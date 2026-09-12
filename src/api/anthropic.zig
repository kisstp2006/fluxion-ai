// SPDX-License-Identifier: BSL-1.0

//! Anthropic's Messages API: Claude, and the servers that copy its shape -
//! DeepSeek has an Anthropic-compatible endpoint too.
//!
//! Three things differ from OpenAI's shape. The system prompt is a field of
//! its own rather than a message. `max_tokens` is required. And a stream is
//! a sequence of named events - a message starts, blocks of content start,
//! grow and stop, the message ends - rather than a list of deltas.
//!
//! Claude reads pictures and does not draw them, so there are no pictures
//! or videos here.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Value = @import("fluxion_json").Value;
const Writer = @import("fluxion_json").Writer;

const Client = @import("../Client.zig");
const ChatStream = @import("../ChatStream.zig");
const chat_types = @import("../chat.zig");
const json = @import("../json.zig");
const sse = @import("../sse.zig");
const transport = @import("../transport.zig");

const ChatRequest = chat_types.ChatRequest;
const Chat = chat_types.Chat;
const Message = chat_types.Message;
const Finish = chat_types.Finish;
const Usage = chat_types.Usage;

/// Sent when a request sets no `max_tokens`, because Anthropic will not
/// answer one without it.
pub const default_max_tokens = 4096;

pub fn chat(c: *Client, a: Allocator, request: ChatRequest, out: *Chat) !void {
    const body = try c.jsonBody(a, request.extra, writeChat, .{ request, false });
    const answer = try c.exchangeJson(a, .{
        .method = .POST,
        .url = try c.endpoint(a, "/messages"),
        .payload = .{ .json = body },
    });
    out.raw = answer.raw;
    parseChat(a, answer.root, out) catch |err| return c.invalid(err, answer.root, "an answer with no content in it");
}

pub fn streamOptions(c: *Client, a: Allocator, request: ChatRequest) !transport.Options {
    const body = try c.jsonBody(a, request.extra, writeChat, .{ request, true });
    return .{
        .method = .POST,
        .url = try c.endpoint(a, "/messages"),
        .payload = .{ .json = body },
        .accept = "text/event-stream",
        .identity = true,
    };
}

fn writeChat(b: *json.Body, request: ChatRequest, stream: bool) !void {
    try b.field("model", request.model);
    try b.field("max_tokens", request.max_tokens orelse default_max_tokens);

    // The system prompt, and any system messages, as one field: a string
    // when there is one piece, a list of text blocks when there are more.
    var system_messages: usize = 0;
    for (request.messages) |message| {
        if (message.role == .system) system_messages += 1;
    }
    if (system_messages == 0) {
        try b.field("system", request.system);
    } else if (try b.key("system")) {
        try b.w.beginArray();
        if (request.system) |system| try textBlock(b, system);
        for (request.messages) |message| {
            if (message.role == .system) try textBlock(b, message.text);
        }
        try b.w.endArray();
    }

    if (try b.key("messages")) {
        try b.w.beginArray();
        for (request.messages) |message| {
            if (message.role == .system) continue;
            try writeMessage(b, message);
        }
        try b.w.endArray();
    }
    try b.field("temperature", request.temperature);
    try b.field("top_p", request.top_p);
    if (request.stop.len > 0) try b.field("stop_sequences", request.stop);
    if (stream) try b.field("stream", true);
}

fn textBlock(b: *json.Body, text: []const u8) !void {
    try b.w.beginObject();
    try b.plain("type", "text");
    try b.plain("text", text);
    try b.w.endObject();
}

fn writeMessage(b: *json.Body, message: Message) !void {
    try b.w.beginObject();
    try b.plain("role", @tagName(message.role));
    if (message.images.len == 0) {
        try b.plain("content", message.text);
    } else {
        try b.w.key("content");
        try b.w.beginArray();
        for (message.images) |image| {
            try b.w.beginObject();
            try b.plain("type", "image");
            try b.w.key("source");
            try b.w.beginObject();
            switch (image) {
                .file => |file| {
                    try b.plain("type", "base64");
                    try b.plain("media_type", file.mime_type);
                    try b.w.key("data");
                    try b.base64(file.bytes);
                },
                .url => |url| {
                    try b.plain("type", "url");
                    try b.plain("url", url);
                },
            }
            try b.w.endObject();
            try b.w.endObject();
        }
        if (message.text.len > 0) try textBlock(b, message.text);
        try b.w.endArray();
    }
    try b.w.endObject();
}

pub fn parseChat(a: Allocator, root: Value, out: *Chat) error{ InvalidResponse, OutOfMemory }!void {
    const content = root.get("content");
    if (content != .array) return error.InvalidResponse;

    var text: std.ArrayList(u8) = .empty;
    var reasoning: std.ArrayList(u8) = .empty;
    for (content.items()) |block| {
        const kind = block.get("type").asString() orelse continue;
        if (std.mem.eql(u8, kind, "text")) {
            try text.appendSlice(a, block.get("text").asString() orelse "");
        } else if (std.mem.eql(u8, kind, "thinking")) {
            if (reasoning.items.len > 0) try reasoning.appendSlice(a, "\n\n");
            try reasoning.appendSlice(a, block.get("thinking").asString() orelse "");
        }
    }
    out.text = text.items;
    out.reasoning = reasoning.items;
    out.finish_reason = root.get("stop_reason").asString() orelse "";
    out.finish = finishOf(out.finish_reason);
    out.usage = .{
        .input_tokens = inputTokens(root.get("usage")),
        .output_tokens = json.count(root.at("/usage/output_tokens")),
    };
    out.model = root.get("model").asString() orelse "";
    out.id = root.get("id").asString() orelse "";
}

/// Anthropic counts cached input apart from the rest. Everything that was
/// read is the sum.
fn inputTokens(usage: Value) ?u64 {
    const plain = json.count(usage.get("input_tokens")) orelse return null;
    return plain +
        (json.count(usage.get("cache_creation_input_tokens")) orelse 0) +
        (json.count(usage.get("cache_read_input_tokens")) orelse 0);
}

pub fn finishOf(reason: []const u8) Finish {
    if (reason.len == 0) return .unknown;
    const table = [_]struct { []const u8, Finish }{
        .{ "end_turn", .stop },
        .{ "stop_sequence", .stop },
        .{ "max_tokens", .length },
        .{ "model_context_window_exceeded", .length },
        .{ "tool_use", .tool_use },
        .{ "refusal", .content_filter },
    };
    for (table) |entry| if (std.mem.eql(u8, reason, entry[0])) return entry[1];
    return .other;
}

pub fn streamEvent(s: *ChatStream, a: Allocator, event: sse.Event) !void {
    const data = std.mem.trim(u8, event.data, " \t\r\n");
    if (data.len == 0) return;
    const root = json.parse(a, data) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        error.InvalidResponse => return s.fail(error.InvalidResponse, "a stream event that is not JSON: {s}", .{data}),
    };
    const kind = root.get("type").asString() orelse event.name;
    const eql = std.mem.eql;

    if (eql(u8, kind, "message_start")) {
        const message = root.get("message");
        if (message.get("id").asString()) |id| try s.setId(id);
        if (message.get("model").asString()) |model| try s.setModel(model);
        s.usage.input_tokens = inputTokens(message.get("usage"));
        s.usage.output_tokens = json.count(message.at("/usage/output_tokens"));
    } else if (eql(u8, kind, "content_block_start")) {
        const block = root.get("content_block");
        const block_kind = block.get("type").asString() orelse "";
        if (eql(u8, block_kind, "text")) {
            const text = block.get("text").asString() orelse "";
            if (text.len > 0) try s.emit(.{ .text = text });
        } else if (eql(u8, block_kind, "thinking")) {
            const thinking = block.get("thinking").asString() orelse "";
            if (thinking.len > 0) try s.emit(.{ .reasoning = thinking });
        }
    } else if (eql(u8, kind, "content_block_delta")) {
        const delta = root.get("delta");
        const delta_kind = delta.get("type").asString() orelse "";
        if (eql(u8, delta_kind, "text_delta")) {
            const text = delta.get("text").asString() orelse "";
            if (text.len > 0) try s.emit(.{ .text = text });
        } else if (eql(u8, delta_kind, "thinking_delta")) {
            const thinking = delta.get("thinking").asString() orelse "";
            if (thinking.len > 0) try s.emit(.{ .reasoning = thinking });
        }
    } else if (eql(u8, kind, "message_delta")) {
        if (root.at("/delta/stop_reason").asString()) |reason| try s.setFinish(reason, finishOf(reason));
        const usage = root.get("usage");
        if (json.count(usage.get("output_tokens"))) |n| s.usage.output_tokens = n;
        if (inputTokens(usage)) |n| s.usage.input_tokens = n;
    } else if (eql(u8, kind, "message_stop")) {
        s.done = true;
    } else if (eql(u8, kind, "error")) {
        const error_kind = root.at("/error/type").asString() orelse "";
        return s.fail(errorOfKind(error_kind), "{s}", .{json.errorMessage(root) orelse data});
    }
    // `ping`, `content_block_stop`, signatures and tool input: nothing to hand on.
}

/// The error a mid-stream `error` event stands for, by its `type`: the same
/// failures a status code would have named, had the stream not already begun.
fn errorOfKind(kind: []const u8) transport.Error {
    const table = [_]struct { []const u8, transport.Error }{
        .{ "overloaded_error", error.ServerError },
        .{ "api_error", error.ServerError },
        .{ "rate_limit_error", error.RateLimited },
        .{ "invalid_request_error", error.BadRequest },
        .{ "request_too_large", error.BadRequest },
        .{ "authentication_error", error.Unauthorized },
        .{ "permission_error", error.Forbidden },
        .{ "not_found_error", error.NotFound },
        .{ "billing_error", error.OutOfCredit },
    };
    for (table) |entry| if (std.mem.eql(u8, kind, entry[0])) return entry[1];
    return error.GenerationFailed;
}

pub fn models(c: *Client, a: Allocator, out: *Client.Models) !void {
    const answer = try c.exchangeJson(a, .{ .url = try c.endpoint(a, "/models?limit=1000") });
    out.raw = answer.raw;
    var list: std.ArrayList(Client.Model) = .empty;
    for (answer.root.get("data").items()) |item| {
        const id = item.get("id").asString() orelse continue;
        try list.append(a, .{ .id = id, .name = item.get("display_name").asString() orelse id });
    }
    out.items = list.items;
}

test "the messages body: system apart, pictures before words, max_tokens always" {
    const gpa = std.testing.allocator;
    var out: std.Io.Writer.Allocating = .init(gpa);
    defer out.deinit();
    var w: Writer = .init(&out.writer, .{});
    var b: json.Body = try .begin(&w, null);
    try writeChat(&b, .{
        .model = "claude-sonnet-5",
        .system = "Be brief.",
        .messages = &.{
            .system("Answer in Hungarian."),
            .{ .role = .user, .text = "What is this?", .images = &.{.fromUrl("https://example.com/cat.png")} },
            .assistant("Egy macska."),
            .user("Biztos?"),
        },
        .stop = &.{"\n\n"},
    }, true);
    try b.end();
    try std.testing.expectEqualStrings(
        \\{"model":"claude-sonnet-5","max_tokens":4096,"system":[{"type":"text","text":"Be brief."},{"type":"text","text":"Answer in Hungarian."}],"messages":[{"role":"user","content":[{"type":"image","source":{"type":"url","url":"https://example.com/cat.png"}},{"type":"text","text":"What is this?"}]},{"role":"assistant","content":"Egy macska."},{"role":"user","content":"Biztos?"}],"stop_sequences":["\n\n"],"stream":true}
    , out.written());
}

test "parse an answer with thinking, and cached input" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const root = try json.parse(a,
        \\{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-5",
        \\"content":[{"type":"thinking","thinking":"Let me see.","signature":"x"},{"type":"text","text":"Hello"},{"type":"text","text":" there"}],
        \\"stop_reason":"end_turn","usage":{"input_tokens":10,"cache_read_input_tokens":90,"output_tokens":5}}
    );
    var out: Chat = .{ .arena = undefined };
    try parseChat(a, root, &out);
    try std.testing.expectEqualStrings("Hello there", out.text);
    try std.testing.expectEqualStrings("Let me see.", out.reasoning);
    try std.testing.expectEqual(Finish.stop, out.finish);
    try std.testing.expectEqual(@as(?u64, 100), out.usage.input_tokens);
    try std.testing.expectEqual(@as(?u64, 5), out.usage.output_tokens);
}
