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
const Value = std.json.Value;

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
        try b.s.beginArray();
        if (request.system) |system| try textBlock(b, system);
        for (request.messages) |message| {
            if (message.role == .system) try textBlock(b, message.text);
        }
        try b.s.endArray();
    }

    if (try b.key("messages")) {
        try b.s.beginArray();
        for (request.messages) |message| {
            if (message.role == .system) continue;
            try writeMessage(b, message);
        }
        try b.s.endArray();
    }
    try b.field("temperature", request.temperature);
    try b.field("top_p", request.top_p);
    if (request.stop.len > 0) try b.field("stop_sequences", request.stop);
    if (stream) try b.field("stream", true);
}

fn textBlock(b: *json.Body, text: []const u8) !void {
    try b.s.beginObject();
    try b.plain("type", "text");
    try b.plain("text", text);
    try b.s.endObject();
}

fn writeMessage(b: *json.Body, message: Message) !void {
    try b.s.beginObject();
    try b.plain("role", @tagName(message.role));
    if (message.images.len == 0) {
        try b.plain("content", message.text);
    } else {
        try b.s.objectField("content");
        try b.s.beginArray();
        for (message.images) |image| {
            try b.s.beginObject();
            try b.plain("type", "image");
            try b.s.objectField("source");
            try b.s.beginObject();
            switch (image) {
                .file => |file| {
                    try b.plain("type", "base64");
                    try b.plain("media_type", file.mime_type);
                    try b.s.objectField("data");
                    try b.base64(file.bytes);
                },
                .url => |url| {
                    try b.plain("type", "url");
                    try b.plain("url", url);
                },
            }
            try b.s.endObject();
            try b.s.endObject();
        }
        if (message.text.len > 0) try textBlock(b, message.text);
        try b.s.endArray();
    }
    try b.s.endObject();
}

pub fn parseChat(a: Allocator, root: Value, out: *Chat) error{ InvalidResponse, OutOfMemory }!void {
    const content = json.at(root, .{"content"}) orelse return error.InvalidResponse;
    if (content != .array) return error.InvalidResponse;

    var text: std.ArrayList(u8) = .empty;
    var reasoning: std.ArrayList(u8) = .empty;
    for (content.array.items) |block| {
        const kind = json.string(json.at(block, .{"type"})) orelse continue;
        if (std.mem.eql(u8, kind, "text")) {
            try text.appendSlice(a, json.string(json.at(block, .{"text"})) orelse "");
        } else if (std.mem.eql(u8, kind, "thinking")) {
            if (reasoning.items.len > 0) try reasoning.appendSlice(a, "\n\n");
            try reasoning.appendSlice(a, json.string(json.at(block, .{"thinking"})) orelse "");
        }
    }
    out.text = text.items;
    out.reasoning = reasoning.items;
    out.finish_reason = json.string(json.at(root, .{"stop_reason"})) orelse "";
    out.finish = finishOf(out.finish_reason);
    out.usage = .{
        .input_tokens = inputTokens(json.at(root, .{"usage"})),
        .output_tokens = json.count(json.at(root, .{ "usage", "output_tokens" })),
    };
    out.model = json.string(json.at(root, .{"model"})) orelse "";
    out.id = json.string(json.at(root, .{"id"})) orelse "";
}

/// Anthropic counts cached input apart from the rest. Everything that was
/// read is the sum.
fn inputTokens(usage: ?Value) ?u64 {
    const plain = json.count(json.at(usage, .{"input_tokens"})) orelse return null;
    return plain +
        (json.count(json.at(usage, .{"cache_creation_input_tokens"})) orelse 0) +
        (json.count(json.at(usage, .{"cache_read_input_tokens"})) orelse 0);
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
    const kind = json.string(json.at(root, .{"type"})) orelse event.name;
    const eql = std.mem.eql;

    if (eql(u8, kind, "message_start")) {
        const message = json.at(root, .{"message"});
        if (json.string(json.at(message, .{"id"}))) |id| try s.setId(id);
        if (json.string(json.at(message, .{"model"}))) |model| try s.setModel(model);
        s.usage.input_tokens = inputTokens(json.at(message, .{"usage"}));
        s.usage.output_tokens = json.count(json.at(message, .{ "usage", "output_tokens" }));
    } else if (eql(u8, kind, "content_block_start")) {
        const block = json.at(root, .{"content_block"});
        const block_kind = json.string(json.at(block, .{"type"})) orelse "";
        if (eql(u8, block_kind, "text")) {
            const text = json.string(json.at(block, .{"text"})) orelse "";
            if (text.len > 0) try s.emit(.{ .text = text });
        } else if (eql(u8, block_kind, "thinking")) {
            const thinking = json.string(json.at(block, .{"thinking"})) orelse "";
            if (thinking.len > 0) try s.emit(.{ .reasoning = thinking });
        }
    } else if (eql(u8, kind, "content_block_delta")) {
        const delta = json.at(root, .{"delta"});
        const delta_kind = json.string(json.at(delta, .{"type"})) orelse "";
        if (eql(u8, delta_kind, "text_delta")) {
            const text = json.string(json.at(delta, .{"text"})) orelse "";
            if (text.len > 0) try s.emit(.{ .text = text });
        } else if (eql(u8, delta_kind, "thinking_delta")) {
            const thinking = json.string(json.at(delta, .{"thinking"})) orelse "";
            if (thinking.len > 0) try s.emit(.{ .reasoning = thinking });
        }
    } else if (eql(u8, kind, "message_delta")) {
        if (json.string(json.at(root, .{ "delta", "stop_reason" }))) |reason| try s.setFinish(reason, finishOf(reason));
        const usage = json.at(root, .{"usage"});
        if (json.count(json.at(usage, .{"output_tokens"}))) |n| s.usage.output_tokens = n;
        if (inputTokens(usage)) |n| s.usage.input_tokens = n;
    } else if (eql(u8, kind, "message_stop")) {
        s.done = true;
    } else if (eql(u8, kind, "error")) {
        const error_kind = json.string(json.at(root, .{ "error", "type" })) orelse "";
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
    for (json.array(json.at(answer.root, .{"data"}))) |item| {
        const id = json.string(json.at(item, .{"id"})) orelse continue;
        try list.append(a, .{ .id = id, .name = json.string(json.at(item, .{"display_name"})) orelse id });
    }
    out.items = list.items;
}

test "the messages body: system apart, pictures before words, max_tokens always" {
    const gpa = std.testing.allocator;
    var out: std.Io.Writer.Allocating = .init(gpa);
    defer out.deinit();
    var b: json.Body = try .begin(&out.writer, null);
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
