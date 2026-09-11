// SPDX-License-Identifier: BSL-1.0

//! OpenAI's API, and everybody's copy of it.
//!
//! Chat completions rather than the newer Responses API: completions are the
//! shape DeepSeek, xAI, Groq, Mistral, OpenRouter, Together, Fireworks,
//! Perplexity, Ollama and LM Studio all took, and one adapter for all of
//! them is the point. A model that only answers at `/responses` is a
//! `Client.call` away.
//!
//! The copies differ in small ways, and the parsing here takes all of them:
//! DeepSeek's `reasoning_content` and OpenRouter's `reasoning`, a `content`
//! that is an array of parts rather than a string, pictures in a chat
//! answer, links where OpenAI would send bytes.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Value = std.json.Value;

const Client = @import("../Client.zig");
const Provider = @import("../Provider.zig");
const ChatStream = @import("../ChatStream.zig");
const chat_types = @import("../chat.zig");
const image_types = @import("../image.zig");
const video_types = @import("../video.zig");
const json = @import("../json.zig");
const media = @import("../media.zig");
const sse = @import("../sse.zig");
const transport = @import("../transport.zig");

const ChatRequest = chat_types.ChatRequest;
const Chat = chat_types.Chat;
const Message = chat_types.Message;
const Finish = chat_types.Finish;
const Usage = chat_types.Usage;
const ImageRequest = image_types.ImageRequest;
const Images = image_types.Images;
const GeneratedImage = image_types.GeneratedImage;
const VideoRequest = video_types.VideoRequest;
const Video = video_types.Video;

// ---------------------------------------------------------------------------
// Chat
// ---------------------------------------------------------------------------

pub fn chat(c: *Client, a: Allocator, request: ChatRequest, out: *Chat) !void {
    const body = try c.jsonBody(a, request.extra, writeChat, .{ &c.provider, request, false });
    const answer = try c.exchangeJson(a, .{
        .method = .POST,
        .url = try c.endpoint(a, "/chat/completions"),
        .payload = .{ .json = body },
    });
    out.raw = answer.raw;
    parseChat(a, answer.root, out) catch |err| return c.invalid(err, answer.root, "an answer with no choices in it");
}

pub fn streamOptions(c: *Client, a: Allocator, request: ChatRequest) !transport.Options {
    const body = try c.jsonBody(a, request.extra, writeChat, .{ &c.provider, request, true });
    return .{
        .method = .POST,
        .url = try c.endpoint(a, "/chat/completions"),
        .payload = .{ .json = body },
        .accept = "text/event-stream",
        .identity = true,
    };
}

fn writeChat(b: *json.Body, provider: *const Provider, request: ChatRequest, stream: bool) !void {
    try b.field("model", request.model);
    if (try b.key("messages")) {
        try b.s.beginArray();
        if (request.system) |system| try writeMessage(b, .system(system));
        for (request.messages) |message| try writeMessage(b, message);
        try b.s.endArray();
    }
    try b.field(@tagName(provider.max_tokens_field), request.max_tokens);
    try b.field("temperature", request.temperature);
    try b.field("top_p", request.top_p);
    if (request.stop.len > 0) try b.field("stop", request.stop);
    if (stream) {
        try b.field("stream", true);
        if (provider.stream_usage and try b.key("stream_options")) {
            try b.s.beginObject();
            try b.plain("include_usage", true);
            try b.s.endObject();
        }
    }
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
            try b.plain("type", "image_url");
            try b.s.objectField("image_url");
            try b.s.beginObject();
            try b.s.objectField("url");
            switch (image) {
                .file => |file| try b.dataUrl(file.mime_type, file.bytes),
                .url => |url| try b.s.write(url),
            }
            try b.s.endObject();
            try b.s.endObject();
        }
        if (message.text.len > 0) {
            try b.s.beginObject();
            try b.plain("type", "text");
            try b.plain("text", message.text);
            try b.s.endObject();
        }
        try b.s.endArray();
    }
    try b.s.endObject();
}

pub fn parseChat(a: Allocator, root: Value, out: *Chat) error{ InvalidResponse, OutOfMemory }!void {
    const choice = json.at(root, .{ "choices", 0 }) orelse return error.InvalidResponse;
    const message = json.at(choice, .{"message"});
    out.text = try contentText(a, json.at(message, .{"content"}));
    out.reasoning = reasoningText(message) orelse "";
    out.images = try messageImages(a, json.at(message, .{"images"}));
    out.finish_reason = json.string(json.at(choice, .{"finish_reason"})) orelse "";
    out.finish = finishOf(out.finish_reason);
    out.usage = usageOf(json.at(root, .{"usage"}));
    out.model = json.string(json.at(root, .{"model"})) orelse "";
    out.id = json.string(json.at(root, .{"id"})) orelse "";
}

/// DeepSeek calls it `reasoning_content`, OpenRouter and some others
/// `reasoning`.
fn reasoningText(message: ?Value) ?[]const u8 {
    return json.string(json.at(message, .{"reasoning_content"})) orelse
        json.string(json.at(message, .{"reasoning"}));
}

/// `content` is a string, or - from a few servers - an array of parts.
fn contentText(a: Allocator, content: ?Value) Allocator.Error![]const u8 {
    if (json.string(content)) |s| return s;
    var text: std.ArrayList(u8) = .empty;
    for (json.array(content)) |part| {
        if (json.string(json.at(part, .{"text"}))) |t| try text.appendSlice(a, t);
    }
    return text.items;
}

/// Pictures in a chat answer, the way OpenRouter sends a drawing model's:
/// `images: [{ "image_url": { "url": "data:image/png;base64,..." } }]`.
fn messageImages(a: Allocator, value: ?Value) error{ InvalidResponse, OutOfMemory }![]const GeneratedImage {
    const items = json.array(value);
    if (items.len == 0) return &.{};
    var list: std.ArrayList(GeneratedImage) = .empty;
    for (items) |item| {
        const url = json.string(json.at(item, .{ "image_url", "url" })) orelse
            json.string(json.at(item, .{"url"})) orelse continue;
        try list.append(a, try imageFromUrl(a, url));
    }
    return list.items;
}

/// A data URL is the picture itself; any other URL is where to fetch it.
fn imageFromUrl(a: Allocator, url: []const u8) error{ InvalidResponse, OutOfMemory }!GeneratedImage {
    if (media.DataUrl.parse(url)) |data| {
        const bytes = try decode(a, data.base64);
        return .{ .bytes = bytes, .mime_type = media.sniff(bytes) orelse data.mime_type };
    }
    return .{ .url = url };
}

fn decode(a: Allocator, base64: []const u8) error{ InvalidResponse, OutOfMemory }![]u8 {
    return media.decodeBase64(a, base64) catch |err| switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        error.InvalidBase64 => error.InvalidResponse,
    };
}

pub fn finishOf(reason: []const u8) Finish {
    if (reason.len == 0) return .unknown;
    const table = [_]struct { []const u8, Finish }{
        .{ "stop", .stop },
        .{ "length", .length },
        .{ "content_filter", .content_filter },
        .{ "tool_calls", .tool_use },
        .{ "function_call", .tool_use },
    };
    for (table) |entry| if (std.mem.eql(u8, reason, entry[0])) return entry[1];
    return .other;
}

fn usageOf(v: ?Value) Usage {
    return .{
        .input_tokens = json.count(json.at(v, .{"prompt_tokens"})) orelse json.count(json.at(v, .{"input_tokens"})),
        .output_tokens = json.count(json.at(v, .{"completion_tokens"})) orelse json.count(json.at(v, .{"output_tokens"})),
    };
}

pub fn streamEvent(s: *ChatStream, a: Allocator, event: sse.Event) !void {
    const data = std.mem.trim(u8, event.data, " \t\r\n");
    if (data.len == 0) return;
    if (std.mem.eql(u8, data, "[DONE]")) {
        s.done = true;
        return;
    }
    const root = json.parse(a, data) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        error.InvalidResponse => return s.fail(error.InvalidResponse, "a stream event that is not JSON: {s}", .{data}),
    };
    // OpenRouter, and others relaying a provider, report its failures here,
    // after the 200 has already gone out.
    if (json.at(root, .{"error"}) != null) {
        return s.fail(error.GenerationFailed, "{s}", .{json.errorMessage(root) orelse data});
    }
    if (json.string(json.at(root, .{"model"}))) |model| try s.setModel(model);
    if (json.string(json.at(root, .{"id"}))) |id| try s.setId(id);
    if (json.isObject(json.at(root, .{"usage"}))) s.usage = usageOf(json.at(root, .{"usage"}));

    const choice = json.at(root, .{ "choices", 0 }) orelse return;
    const delta = json.at(choice, .{"delta"});
    if (reasoningText(delta)) |reasoning| if (reasoning.len > 0) try s.emit(.{ .reasoning = reasoning });
    const content = try contentText(a, json.at(delta, .{"content"}));
    if (content.len > 0) try s.emit(.{ .text = content });
    for (json.array(json.at(delta, .{"images"}))) |item| {
        const url = json.string(json.at(item, .{ "image_url", "url" })) orelse continue;
        const image = imageFromUrl(a, url) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            error.InvalidResponse => return s.fail(error.InvalidResponse, "a picture in the stream that does not decode", .{}),
        };
        try s.emitImage(image);
    }
    if (json.string(json.at(choice, .{"finish_reason"}))) |reason| try s.setFinish(reason, finishOf(reason));
}

// ---------------------------------------------------------------------------
// Pictures
// ---------------------------------------------------------------------------

pub fn images(c: *Client, a: Allocator, request: ImageRequest, out: *Images) !void {
    const answer = if (request.references.len == 0) answer: {
        const body = try c.jsonBody(a, request.extra, writeImages, .{request});
        break :answer try c.exchangeJson(a, .{
            .method = .POST,
            .url = try c.endpoint(a, "/images/generations"),
            .payload = .{ .json = body },
        });
    } else answer: {
        // Pictures to start from go as files, in a form, to `/images/edits`.
        var form: transport.Form = .init(a, c.io);
        try form.field("model", request.model);
        try form.field("prompt", request.prompt);
        if (request.count != 1) try form.field("n", try std.fmt.allocPrint(a, "{d}", .{request.count}));
        if (request.size) |size| try form.field("size", size);
        if (request.quality) |quality| try form.field("quality", quality);
        const name = if (request.references.len == 1) "image" else "image[]";
        for (request.references, 0..) |reference, i| {
            const filename = try std.fmt.allocPrint(a, "reference-{d}.{s}", .{ i + 1, reference.extension() });
            try form.file(name, filename, reference.mime_type, reference.bytes);
        }
        try form.fields(try c.extraMembers(a, request.extra));
        break :answer try c.exchangeJson(a, .{
            .method = .POST,
            .url = try c.endpoint(a, "/images/edits"),
            .payload = .{ .form = .{ .content_type = form.contentType(), .bytes = try form.finish() } },
        });
    };
    out.raw = answer.raw;

    var list: std.ArrayList(GeneratedImage) = .empty;
    for (json.array(json.at(answer.root, .{"data"}))) |item| {
        var image: GeneratedImage = undefined;
        if (json.string(json.at(item, .{"b64_json"}))) |b64| {
            const bytes = decode(a, b64) catch |err| return c.invalid(err, answer.root, "a picture that is not base64");
            image = .{ .bytes = bytes, .mime_type = media.sniff(bytes) orelse "image/png" };
        } else if (json.string(json.at(item, .{"url"}))) |url| {
            image = imageFromUrl(a, url) catch |err| return c.invalid(err, answer.root, "a picture that is not base64");
        } else continue;
        image.revised_prompt = json.string(json.at(item, .{"revised_prompt"}));
        try list.append(a, image);
    }
    if (list.items.len == 0) return c.invalid(error.InvalidResponse, answer.root, "an answer with no pictures in it");
    out.images = list.items;
    out.usage = usageOf(json.at(answer.root, .{"usage"}));
}

fn writeImages(b: *json.Body, request: ImageRequest) !void {
    try b.field("model", request.model);
    try b.field("prompt", request.prompt);
    if (request.count != 1) try b.field("n", request.count);
    try b.field("size", request.size);
    try b.field("quality", request.quality);
    try b.field("aspect_ratio", request.aspect_ratio);
}

// ---------------------------------------------------------------------------
// Video
// ---------------------------------------------------------------------------

pub fn startVideo(c: *Client, a: Allocator, request: VideoRequest, out: *Video) !void {
    switch (c.provider.video_style) {
        .openai => try startSora(c, a, request, out),
        .xai => try startXai(c, a, request, out),
    }
}

pub fn videoStatus(c: *Client, a: Allocator, id: []const u8, out: *Video) !void {
    const path = try std.fmt.allocPrint(a, "/videos/{s}", .{try pathSegment(a, id)});
    const answer = try c.exchangeJson(a, .{ .url = try c.endpoint(a, path) });
    out.raw = answer.raw;
    switch (c.provider.video_style) {
        .openai => try parseSora(c, a, answer.root, out),
        .xai => try parseXai(a, answer.root, id, out),
    }
}

fn startSora(c: *Client, a: Allocator, request: VideoRequest, out: *Video) !void {
    const url = try c.endpoint(a, "/videos");
    const first_frame_link = if (request.first_frame) |frame| frame == .url else false;
    const answer = if (first_frame_link) answer: {
        // A link to start from goes in JSON; a file needs the form.
        const body = try c.jsonBody(a, request.extra, writeSora, .{request});
        break :answer try c.exchangeJson(a, .{ .method = .POST, .url = url, .payload = .{ .json = body } });
    } else answer: {
        var form: transport.Form = .init(a, c.io);
        try form.field("model", request.model);
        try form.field("prompt", request.prompt);
        if (request.seconds) |seconds| try form.field("seconds", try std.fmt.allocPrint(a, "{d}", .{seconds}));
        if (request.size) |size| try form.field("size", size);
        if (request.first_frame) |frame| {
            const file = frame.file;
            const filename = try std.fmt.allocPrint(a, "first-frame.{s}", .{file.extension()});
            try form.file("input_reference", filename, file.mime_type, file.bytes);
        }
        try form.fields(try c.extraMembers(a, request.extra));
        break :answer try c.exchangeJson(a, .{
            .method = .POST,
            .url = url,
            .payload = .{ .form = .{ .content_type = form.contentType(), .bytes = try form.finish() } },
        });
    };
    out.raw = answer.raw;
    try parseSora(c, a, answer.root, out);
}

fn writeSora(b: *json.Body, request: VideoRequest) !void {
    try b.field("model", request.model);
    try b.field("prompt", request.prompt);
    if (request.seconds) |seconds| {
        var buffer: [16]u8 = undefined;
        try b.field("seconds", std.fmt.bufPrint(&buffer, "{d}", .{seconds}) catch unreachable);
    }
    try b.field("size", request.size);
    if (try b.key("input_reference")) {
        try b.s.beginObject();
        try b.plain("image_url", request.first_frame.?.url);
        try b.s.endObject();
    }
}

fn parseSora(c: *Client, a: Allocator, root: Value, out: *Video) !void {
    out.id = json.string(json.at(root, .{"id"})) orelse
        return c.invalid(error.InvalidResponse, root, "a video with no id");
    out.status = .fromWord(json.string(json.at(root, .{"status"})) orelse "queued");
    if (json.integer(json.at(root, .{"progress"}))) |progress| out.progress = @intCast(std.math.clamp(progress, 0, 100));
    out.message = json.string(json.at(root, .{ "error", "message" })) orelse
        json.string(json.at(root, .{"error"})) orelse "";
    if (out.status == .completed) {
        out.url = try c.endpoint(a, try std.fmt.allocPrint(a, "/videos/{s}/content", .{try pathSegment(a, out.id)}));
    }
}

fn startXai(c: *Client, a: Allocator, request: VideoRequest, out: *Video) !void {
    const body = try c.jsonBody(a, request.extra, writeXai, .{request});
    const answer = try c.exchangeJson(a, .{
        .method = .POST,
        .url = try c.endpoint(a, "/videos/generations"),
        .payload = .{ .json = body },
    });
    out.raw = answer.raw;
    const id = json.string(json.at(answer.root, .{"request_id"})) orelse
        json.string(json.at(answer.root, .{"id"})) orelse
        return c.invalid(error.InvalidResponse, answer.root, "a video with no request id");
    try parseXai(a, answer.root, id, out);
}

fn writeXai(b: *json.Body, request: VideoRequest) !void {
    try b.field("model", request.model);
    try b.field("prompt", request.prompt);
    try b.field("duration", request.seconds);
    try b.field("aspect_ratio", request.aspect_ratio);
    try b.field("resolution", request.resolution);
    if (request.first_frame) |frame| {
        if (try b.key("image")) {
            try b.s.beginObject();
            try b.s.objectField("url");
            switch (frame) {
                .url => |url| try b.s.write(url),
                .file => |file| try b.dataUrl(file.mime_type, file.bytes),
            }
            try b.s.endObject();
        }
    }
}

fn parseXai(a: Allocator, root: Value, id: []const u8, out: *Video) !void {
    out.id = try a.dupe(u8, id);
    out.url = json.string(json.at(root, .{ "video", "url" })) orelse json.string(json.at(root, .{"url"}));
    out.status = if (json.string(json.at(root, .{"status"}))) |word| .fromWord(word) else if (out.url != null) .completed else .queued;
    if (json.integer(json.at(root, .{"progress"}))) |progress| out.progress = @intCast(std.math.clamp(progress, 0, 100));
    out.message = json.string(json.at(root, .{ "error", "message" })) orelse
        json.string(json.at(root, .{"error"})) orelse "";
}

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------

pub fn models(c: *Client, a: Allocator, out: *Client.Models) !void {
    const answer = try c.exchangeJson(a, .{ .url = try c.endpoint(a, "/models") });
    out.raw = answer.raw;
    var list: std.ArrayList(Client.Model) = .empty;
    for (json.array(json.at(answer.root, .{"data"}))) |item| {
        const id = json.string(json.at(item, .{"id"})) orelse continue;
        try list.append(a, .{ .id = id, .name = json.string(json.at(item, .{"name"})) orelse id });
    }
    out.items = list.items;
}

/// `id` made safe to be one segment of a path.
pub fn pathSegment(a: Allocator, id: []const u8) Allocator.Error![]const u8 {
    for (id) |ch| {
        if (!isUnreserved(ch)) break;
    } else return id;
    var out: std.ArrayList(u8) = .empty;
    for (id) |ch| {
        if (isUnreserved(ch)) {
            try out.append(a, ch);
        } else {
            try out.print(a, "%{X:0>2}", .{ch});
        }
    }
    return out.toOwnedSlice(a);
}

fn isUnreserved(ch: u8) bool {
    return std.ascii.isAlphanumeric(ch) or ch == '-' or ch == '.' or ch == '_' or ch == '~';
}

test finishOf {
    try std.testing.expectEqual(Finish.stop, finishOf("stop"));
    try std.testing.expectEqual(Finish.tool_use, finishOf("tool_calls"));
    try std.testing.expectEqual(Finish.other, finishOf("insufficient_system_resource"));
    try std.testing.expectEqual(Finish.unknown, finishOf(""));
}

test pathSegment {
    const gpa = std.testing.allocator;
    try std.testing.expectEqualStrings("video_68d7", try pathSegment(gpa, "video_68d7"));
    const escaped = try pathSegment(gpa, "a/b c");
    defer gpa.free(escaped);
    try std.testing.expectEqualStrings("a%2Fb%20c", escaped);
}

test "parse a DeepSeek answer with its reasoning" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const root = try json.parse(a,
        \\{"id":"x1","model":"deepseek-flash","choices":[{"index":0,"message":{"role":"assistant",
        \\"content":"4","reasoning_content":"2+2 is 4."},"finish_reason":"stop"}],
        \\"usage":{"prompt_tokens":12,"completion_tokens":7,"total_tokens":19}}
    );
    var out: Chat = .{ .arena = undefined };
    try parseChat(a, root, &out);
    try std.testing.expectEqualStrings("4", out.text);
    try std.testing.expectEqualStrings("2+2 is 4.", out.reasoning);
    try std.testing.expectEqual(Finish.stop, out.finish);
    try std.testing.expectEqual(@as(?u64, 12), out.usage.input_tokens);
    try std.testing.expectEqual(@as(?u64, 7), out.usage.output_tokens);
    try std.testing.expectEqualStrings("deepseek-flash", out.model);
}

test "parse content given as parts, and a picture in the answer" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const root = try json.parse(a,
        \\{"choices":[{"message":{"content":[{"type":"text","text":"Here "},{"type":"text","text":"it is"}],
        \\"images":[{"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgo="}}]},
        \\"finish_reason":"stop"}]}
    );
    var out: Chat = .{ .arena = undefined };
    try parseChat(a, root, &out);
    try std.testing.expectEqualStrings("Here it is", out.text);
    try std.testing.expectEqual(@as(usize, 1), out.images.len);
    try std.testing.expectEqualStrings("image/png", out.images[0].mime_type);
    try std.testing.expectEqualSlices(u8, "\x89PNG\r\n\x1a\n", out.images[0].bytes);
}

test "the chat body, as OpenAI and as a compatible server" {
    const gpa = std.testing.allocator;
    const request: ChatRequest = .{
        .model = "gpt-5-mini",
        .system = "Be brief.",
        .messages = &.{
            .user("What is this?"),
            .{ .role = .user, .text = "And this?", .images = &.{.fromBytes("\x89PNG\r\n\x1a\n")} },
        },
        .max_tokens = 100,
        .temperature = 0.7,
    };
    var out: std.Io.Writer.Allocating = .init(gpa);
    defer out.deinit();
    const openai_provider: Provider = .openai("k");
    const deepseek_provider: Provider = .deepseek("k");
    var b: json.Body = try .begin(&out.writer, null);
    try writeChat(&b, &openai_provider, request, true);
    try b.end();
    try std.testing.expectEqualStrings(
        \\{"model":"gpt-5-mini","messages":[{"role":"system","content":"Be brief."},{"role":"user","content":"What is this?"},{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgo="}},{"type":"text","text":"And this?"}]}],"max_completion_tokens":100,"temperature":0.7,"stream":true,"stream_options":{"include_usage":true}}
    , out.written());

    out.clearRetainingCapacity();
    b = try .begin(&out.writer, null);
    try writeChat(&b, &deepseek_provider, .{ .model = "deepseek-flash", .messages = &.{.user("hi")}, .max_tokens = 5 }, false);
    try b.end();
    try std.testing.expectEqualStrings(
        \\{"model":"deepseek-flash","messages":[{"role":"user","content":"hi"}],"max_tokens":5}
    , out.written());
}
