// SPDX-License-Identifier: BSL-1.0

//! Google's Gemini API: `generateContent` for words and for the image
//! models that draw in conversation, `predict` for Imagen, and
//! `predictLongRunning` for Veo.
//!
//! The model is part of the path rather than the body, a turn is a list of
//! parts - text, inline data, a file - and the assistant is called `model`.
//! The same base URL and a bearer token also reach Vertex AI, whose paths
//! end the same way.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Value = std.json.Value;

const Client = @import("../Client.zig");
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
const Image = chat_types.Image;
const Finish = chat_types.Finish;
const Usage = chat_types.Usage;
const ImageRequest = image_types.ImageRequest;
const Images = image_types.Images;
const GeneratedImage = image_types.GeneratedImage;
const VideoRequest = video_types.VideoRequest;
const Video = video_types.Video;

/// `/models/{model}:{method}`. A model given with its collection already -
/// `models/...`, `tunedModels/...`, a Vertex publisher path - is used as it
/// is.
fn modelPath(a: Allocator, model: []const u8, method: []const u8) Allocator.Error![]const u8 {
    if (std.mem.findScalar(u8, model, '/') != null) return std.fmt.allocPrint(a, "/{s}:{s}", .{ model, method });
    return std.fmt.allocPrint(a, "/models/{s}:{s}", .{ model, method });
}

// ---------------------------------------------------------------------------
// Chat
// ---------------------------------------------------------------------------

pub fn chat(c: *Client, a: Allocator, request: ChatRequest, out: *Chat) !void {
    const body = try c.jsonBody(a, request.extra, writeChat, .{request});
    const answer = try c.exchangeJson(a, .{
        .method = .POST,
        .url = try c.endpoint(a, try modelPath(a, request.model, "generateContent")),
        .payload = .{ .json = body },
    });
    out.raw = answer.raw;

    var harvest: Harvest = .{};
    harvest.gather(a, answer.root) catch |err| return c.invalid(err, answer.root, "an answer that does not parse");
    if (json.at(answer.root, .{"candidates"}) == null and harvest.finish_reason.len == 0)
        return c.invalid(error.InvalidResponse, answer.root, "an answer with no candidates in it");
    out.text = harvest.text.items;
    out.reasoning = harvest.reasoning.items;
    out.images = harvest.images.items;
    out.finish_reason = harvest.finish_reason;
    out.finish = finishOf(harvest.finish_reason);
    out.usage = harvest.usage;
    out.model = harvest.model;
    out.id = harvest.id;
}

pub fn streamOptions(c: *Client, a: Allocator, request: ChatRequest) !transport.Options {
    const body = try c.jsonBody(a, request.extra, writeChat, .{request});
    const path = try std.fmt.allocPrint(a, "{s}?alt=sse", .{try modelPath(a, request.model, "streamGenerateContent")});
    return .{
        .method = .POST,
        .url = try c.endpoint(a, path),
        .payload = .{ .json = body },
        .accept = "text/event-stream",
        .identity = true,
    };
}

fn writeChat(b: *json.Body, request: ChatRequest) !void {
    if (try b.key("contents")) {
        try b.s.beginArray();
        for (request.messages) |message| {
            if (message.role == .system) continue;
            try b.s.beginObject();
            try b.plain("role", if (message.role == .assistant) "model" else "user");
            try b.s.objectField("parts");
            try b.s.beginArray();
            for (message.images) |image| try imagePart(b, image);
            if (message.text.len > 0 or message.images.len == 0) try textPart(b, message.text);
            try b.s.endArray();
            try b.s.endObject();
        }
        try b.s.endArray();
    }

    var has_system = request.system != null;
    for (request.messages) |message| has_system = has_system or message.role == .system;
    if (has_system and try b.key("systemInstruction")) {
        try b.s.beginObject();
        try b.s.objectField("parts");
        try b.s.beginArray();
        if (request.system) |system| try textPart(b, system);
        for (request.messages) |message| {
            if (message.role == .system) try textPart(b, message.text);
        }
        try b.s.endArray();
        try b.s.endObject();
    }

    const configured = request.max_tokens != null or request.temperature != null or
        request.top_p != null or request.stop.len > 0;
    if (configured and try b.key("generationConfig")) {
        try b.s.beginObject();
        try b.plain("maxOutputTokens", request.max_tokens);
        try b.plain("temperature", request.temperature);
        try b.plain("topP", request.top_p);
        if (request.stop.len > 0) try b.plain("stopSequences", request.stop);
        try b.s.endObject();
    }
}

fn textPart(b: *json.Body, text: []const u8) !void {
    try b.s.beginObject();
    try b.plain("text", text);
    try b.s.endObject();
}

fn imagePart(b: *json.Body, image: Image) !void {
    try b.s.beginObject();
    switch (image) {
        .file => |file| try inlineData(b, file),
        .url => |url| {
            try b.s.objectField("fileData");
            try b.s.beginObject();
            try b.plain("fileUri", url);
            try b.s.endObject();
        },
    }
    try b.s.endObject();
}

/// `"inlineData": {"mimeType": ..., "data": ...}` - the field and its
/// value, inside an object the caller opened.
fn inlineData(b: *json.Body, file: media.Media) !void {
    try b.s.objectField("inlineData");
    try b.s.beginObject();
    try b.plain("mimeType", file.mime_type);
    try b.s.objectField("data");
    try b.base64(file.bytes);
    try b.s.endObject();
}

/// What one `GenerateContentResponse` holds, gathered from the first
/// candidate: words, thoughts, pictures, and why it ended.
const Harvest = struct {
    text: std.ArrayList(u8) = .empty,
    reasoning: std.ArrayList(u8) = .empty,
    images: std.ArrayList(GeneratedImage) = .empty,
    finish_reason: []const u8 = "",
    usage: Usage = .{},
    model: []const u8 = "",
    id: []const u8 = "",

    fn gather(h: *Harvest, a: Allocator, root: Value) error{ InvalidResponse, OutOfMemory }!void {
        const candidate = json.at(root, .{ "candidates", 0 });
        for (json.array(json.at(candidate, .{ "content", "parts" }))) |part| {
            if (json.string(json.at(part, .{"text"}))) |text| {
                const thought = json.boolean(json.at(part, .{"thought"})) orelse false;
                try (if (thought) &h.reasoning else &h.text).appendSlice(a, text);
            } else if (json.at(part, .{"inlineData"}) orelse json.at(part, .{"inline_data"})) |data| {
                const b64 = json.string(json.at(data, .{"data"})) orelse continue;
                const bytes = media.decodeBase64(a, b64) catch |err| switch (err) {
                    error.OutOfMemory => return error.OutOfMemory,
                    error.InvalidBase64 => return error.InvalidResponse,
                };
                const claimed = json.string(json.at(data, .{"mimeType"})) orelse json.string(json.at(data, .{"mime_type"}));
                try h.images.append(a, .{ .bytes = bytes, .mime_type = media.sniff(bytes) orelse claimed orelse "application/octet-stream" });
            }
        }
        if (json.string(json.at(candidate, .{"finishReason"}))) |reason| h.finish_reason = reason;
        // Blocked before a word was written: no candidate, and a reason why.
        if (json.string(json.at(root, .{ "promptFeedback", "blockReason" }))) |reason| h.finish_reason = reason;

        const usage = json.at(root, .{"usageMetadata"});
        if (usage != null) {
            h.usage.input_tokens = json.count(json.at(usage, .{"promptTokenCount"}));
            const answer = json.count(json.at(usage, .{"candidatesTokenCount"}));
            const thoughts = json.count(json.at(usage, .{"thoughtsTokenCount"}));
            h.usage.output_tokens = if (answer == null and thoughts == null) null else (answer orelse 0) + (thoughts orelse 0);
        }
        if (json.string(json.at(root, .{"modelVersion"}))) |model| h.model = model;
        if (json.string(json.at(root, .{"responseId"}))) |id| h.id = id;
    }
};

pub fn finishOf(reason: []const u8) Finish {
    if (reason.len == 0 or std.mem.eql(u8, reason, "FINISH_REASON_UNSPECIFIED")) return .unknown;
    if (std.mem.eql(u8, reason, "STOP")) return .stop;
    if (std.mem.eql(u8, reason, "MAX_TOKENS")) return .length;
    if (std.mem.eql(u8, reason, "MALFORMED_FUNCTION_CALL") or std.mem.eql(u8, reason, "UNEXPECTED_TOOL_CALL")) return .tool_use;
    const filtered = [_][]const u8{
        "SAFETY",       "RECITATION",               "BLOCKLIST",        "PROHIBITED_CONTENT", "SPII",
        "IMAGE_SAFETY", "IMAGE_PROHIBITED_CONTENT", "IMAGE_RECITATION", "OTHER_SAFETY",
    };
    for (filtered) |word| if (std.mem.eql(u8, reason, word)) return .content_filter;
    return .other;
}

pub fn streamEvent(s: *ChatStream, a: Allocator, event: sse.Event) !void {
    const data = std.mem.trim(u8, event.data, " \t\r\n");
    if (data.len == 0) return;
    const root = json.parse(a, data) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        error.InvalidResponse => return s.fail(error.InvalidResponse, "a stream event that is not JSON: {s}", .{data}),
    };
    if (json.at(root, .{"error"}) != null) {
        return s.fail(error.GenerationFailed, "{s}", .{json.errorMessage(root) orelse data});
    }
    var harvest: Harvest = .{};
    harvest.gather(a, root) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        error.InvalidResponse => return s.fail(error.InvalidResponse, "a picture in the stream that does not decode", .{}),
    };
    if (harvest.reasoning.items.len > 0) try s.emit(.{ .reasoning = harvest.reasoning.items });
    if (harvest.text.items.len > 0) try s.emit(.{ .text = harvest.text.items });
    for (harvest.images.items) |image| try s.emitImage(image);
    if (harvest.finish_reason.len > 0) try s.setFinish(harvest.finish_reason, finishOf(harvest.finish_reason));
    if (harvest.usage.input_tokens != null or harvest.usage.output_tokens != null) s.usage = harvest.usage;
    if (harvest.model.len > 0) try s.setModel(harvest.model);
    if (harvest.id.len > 0) try s.setId(harvest.id);
}

// ---------------------------------------------------------------------------
// Pictures
// ---------------------------------------------------------------------------

fn isImagen(model: []const u8) bool {
    return std.mem.find(u8, model, "imagen") != null;
}

pub fn images(c: *Client, a: Allocator, request: ImageRequest, out: *Images) !void {
    if (isImagen(request.model)) return imagen(c, a, request, out);

    // Gemini's image models draw one picture an answer; more is more asking.
    var list: std.ArrayList(GeneratedImage) = .empty;
    var text: std.ArrayList(u8) = .empty;
    var usage: Usage = .{};
    var last_reason: []const u8 = "";
    const url = try c.endpoint(a, try modelPath(a, request.model, "generateContent"));
    const body = try c.jsonBody(a, request.extra, writeDrawing, .{request});
    for (0..@max(request.count, 1)) |_| {
        const answer = try c.exchangeJson(a, .{ .method = .POST, .url = url, .payload = .{ .json = body } });
        out.raw = answer.raw;
        var harvest: Harvest = .{};
        harvest.gather(a, answer.root) catch |err| return c.invalid(err, answer.root, "a picture that does not decode");
        try list.appendSlice(a, harvest.images.items);
        try text.appendSlice(a, harvest.text.items);
        last_reason = harvest.finish_reason;
        usage.input_tokens = sum(usage.input_tokens, harvest.usage.input_tokens);
        usage.output_tokens = sum(usage.output_tokens, harvest.usage.output_tokens);
    }
    if (list.items.len == 0) {
        // Refused, usually, and the model or the finish reason says why.
        const why = if (text.items.len > 0) text.items else if (last_reason.len > 0) last_reason else "no picture came back";
        return c.fail(0, error.GenerationFailed, "{s}", .{why});
    }
    out.images = list.items;
    out.text = text.items;
    out.usage = usage;
}

fn sum(a: ?u64, b: ?u64) ?u64 {
    if (a == null and b == null) return null;
    return (a orelse 0) + (b orelse 0);
}

fn writeDrawing(b: *json.Body, request: ImageRequest) !void {
    if (try b.key("contents")) {
        try b.s.beginArray();
        try b.s.beginObject();
        try b.plain("role", "user");
        try b.s.objectField("parts");
        try b.s.beginArray();
        for (request.references) |reference| {
            try b.s.beginObject();
            try inlineData(b, reference);
            try b.s.endObject();
        }
        try textPart(b, request.prompt);
        try b.s.endArray();
        try b.s.endObject();
        try b.s.endArray();
    }
    if (try b.key("generationConfig")) {
        try b.s.beginObject();
        try b.plain("responseModalities", &[_][]const u8{ "TEXT", "IMAGE" });
        if (request.aspect_ratio != null or request.size != null) {
            try b.s.objectField("imageConfig");
            try b.s.beginObject();
            try b.plain("aspectRatio", request.aspect_ratio);
            try b.plain("imageSize", request.size);
            try b.s.endObject();
        }
        try b.s.endObject();
    }
}

fn imagen(c: *Client, a: Allocator, request: ImageRequest, out: *Images) !void {
    if (request.references.len > 0) return c.fail(0, error.Unsupported, "Imagen draws from a prompt alone; use a Gemini image model to start from a picture", .{});
    const extra = try c.extraMembers(a, request.extra);
    const body = try c.jsonBody(a, null, writeImagen, .{ request, extra });
    const answer = try c.exchangeJson(a, .{
        .method = .POST,
        .url = try c.endpoint(a, try modelPath(a, request.model, "predict")),
        .payload = .{ .json = body },
    });
    out.raw = answer.raw;
    var list: std.ArrayList(GeneratedImage) = .empty;
    var filtered: ?[]const u8 = null;
    for (json.array(json.at(answer.root, .{"predictions"}))) |prediction| {
        const b64 = json.string(json.at(prediction, .{"bytesBase64Encoded"})) orelse {
            filtered = json.string(json.at(prediction, .{"raiFilteredReason"})) orelse filtered;
            continue;
        };
        const bytes = media.decodeBase64(a, b64) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            error.InvalidBase64 => return c.invalid(error.InvalidResponse, answer.root, "a picture that is not base64"),
        };
        const claimed = json.string(json.at(prediction, .{"mimeType"}));
        try list.append(a, .{ .bytes = bytes, .mime_type = media.sniff(bytes) orelse claimed orelse "image/png" });
    }
    if (list.items.len == 0) {
        return c.fail(0, error.GenerationFailed, "{s}", .{filtered orelse "no picture came back; the prompt may have been filtered"});
    }
    out.images = list.items;
}

/// Imagen's `predict`: the prompt as the one instance, and everything else,
/// `extra` included, as parameters.
fn writeImagen(b: *json.Body, request: ImageRequest, extra: ?std.json.ObjectMap) !void {
    try b.s.objectField("instances");
    try b.s.beginArray();
    try b.s.beginObject();
    try b.plain("prompt", request.prompt);
    try b.s.endObject();
    try b.s.endArray();

    try b.s.objectField("parameters");
    try b.s.beginWriteRaw();
    var parameters: json.Body = try .begin(b.s.writer, extra);
    try parameters.field("sampleCount", request.count);
    try parameters.field("aspectRatio", request.aspect_ratio);
    try parameters.field("imageSize", request.size);
    try parameters.end();
    b.s.endWriteRaw();
}

// ---------------------------------------------------------------------------
// Video
// ---------------------------------------------------------------------------

pub fn startVideo(c: *Client, a: Allocator, request: VideoRequest, out: *Video) !void {
    if (request.first_frame) |frame| {
        if (frame == .url) return c.fail(0, error.Unsupported, "Veo takes the first frame as a file, not a link", .{});
    }
    const extra = try c.extraMembers(a, request.extra);
    const body = try c.jsonBody(a, null, writeVeo, .{ request, extra });
    const answer = try c.exchangeJson(a, .{
        .method = .POST,
        .url = try c.endpoint(a, try modelPath(a, request.model, "predictLongRunning")),
        .payload = .{ .json = body },
    });
    out.raw = answer.raw;
    try parseOperation(c, answer.root, out);
}

fn writeVeo(b: *json.Body, request: VideoRequest, extra: ?std.json.ObjectMap) !void {
    try b.s.objectField("instances");
    try b.s.beginArray();
    try b.s.beginObject();
    try b.plain("prompt", request.prompt);
    if (request.first_frame) |frame| {
        try b.s.objectField("image");
        try b.s.beginObject();
        try inlineData(b, frame.file);
        try b.s.endObject();
    }
    try b.s.endObject();
    try b.s.endArray();

    try b.s.objectField("parameters");
    try b.s.beginWriteRaw();
    var parameters: json.Body = try .begin(b.s.writer, extra);
    try parameters.field("aspectRatio", request.aspect_ratio);
    try parameters.field("durationSeconds", request.seconds);
    try parameters.field("resolution", request.resolution);
    try parameters.field("negativePrompt", request.negative_prompt);
    try parameters.end();
    b.s.endWriteRaw();
}

pub fn videoStatus(c: *Client, a: Allocator, id: []const u8, out: *Video) !void {
    // The id is the operation's name, which is already a path.
    const answer = try c.exchangeJson(a, .{ .url = try c.endpoint(a, id) });
    out.raw = answer.raw;
    try parseOperation(c, answer.root, out);
}

fn parseOperation(c: *Client, root: Value, out: *Video) !void {
    out.id = json.string(json.at(root, .{"name"})) orelse
        return c.invalid(error.InvalidResponse, root, "an operation with no name");
    const done = json.boolean(json.at(root, .{"done"})) orelse false;
    if (json.at(root, .{"error"}) != null) {
        out.status = .failed;
        out.message = json.errorMessage(root) orelse "the operation failed";
        return;
    }
    if (!done) {
        out.status = if (json.at(root, .{"metadata"}) != null) .in_progress else .queued;
        return;
    }
    const response = json.at(root, .{"response"});
    const uri = json.string(json.at(response, .{ "generateVideoResponse", "generatedSamples", 0, "video", "uri" })) orelse
        json.string(json.at(response, .{ "generatedVideos", 0, "video", "uri" })) orelse
        json.string(json.at(response, .{ "videos", 0, "uri" }));
    if (uri) |u| {
        out.status = .completed;
        out.url = u;
    } else {
        out.status = .failed;
        out.message = json.string(json.at(response, .{ "generateVideoResponse", "raiMediaFilteredReasons", 0 })) orelse
            "the operation finished without a video";
    }
}

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------

pub fn models(c: *Client, a: Allocator, out: *Client.Models) !void {
    const answer = try c.exchangeJson(a, .{ .url = try c.endpoint(a, "/models?pageSize=1000") });
    out.raw = answer.raw;
    var list: std.ArrayList(Client.Model) = .empty;
    for (json.array(json.at(answer.root, .{"models"}))) |item| {
        const name = json.string(json.at(item, .{"name"})) orelse continue;
        // `models/gemini-...`: the part after the slash is what a request names.
        const id = if (std.mem.startsWith(u8, name, "models/")) name["models/".len..] else name;
        try list.append(a, .{ .id = id, .name = json.string(json.at(item, .{"displayName"})) orelse id });
    }
    out.items = list.items;
}

test "the generateContent body" {
    const gpa = std.testing.allocator;
    var out: std.Io.Writer.Allocating = .init(gpa);
    defer out.deinit();
    var b: json.Body = try .begin(&out.writer, null);
    try writeChat(&b, .{
        .model = "gemini-3.5-flash",
        .system = "Be brief.",
        .messages = &.{
            .{ .role = .user, .text = "What is this?", .images = &.{.fromBytes("\xff\xd8\xff\xe0")} },
            .assistant("A photo."),
        },
        .max_tokens = 64,
    });
    try b.end();
    try std.testing.expectEqualStrings(
        \\{"contents":[{"role":"user","parts":[{"inlineData":{"mimeType":"image/jpeg","data":"/9j/4A=="}},{"text":"What is this?"}]},{"role":"model","parts":[{"text":"A photo."}]}],"systemInstruction":{"parts":[{"text":"Be brief."}]},"generationConfig":{"maxOutputTokens":64}}
    , out.written());
}

test "Imagen and Veo put extra into parameters" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var out: std.Io.Writer.Allocating = .init(gpa);
    defer out.deinit();

    const extra = try json.parseExtra(arena.allocator(), "{\"personGeneration\":\"allow_adult\"}");
    var b: json.Body = try .begin(&out.writer, null);
    try writeImagen(&b, .{ .model = "imagen-4.0-generate-001", .prompt = "a fox", .count = 2, .aspect_ratio = "16:9" }, extra);
    try b.end();
    try std.testing.expectEqualStrings(
        \\{"instances":[{"prompt":"a fox"}],"parameters":{"sampleCount":2,"aspectRatio":"16:9","personGeneration":"allow_adult"}}
    , out.written());

    out.clearRetainingCapacity();
    b = try .begin(&out.writer, null);
    try writeVeo(&b, .{ .model = "veo-3.1-generate-preview", .prompt = "waves", .seconds = 8, .first_frame = .fromBytes("\x89PNG\r\n\x1a\n") }, null);
    try b.end();
    try std.testing.expectEqualStrings(
        \\{"instances":[{"prompt":"waves","image":{"inlineData":{"mimeType":"image/png","data":"iVBORw0KGgo="}}}],"parameters":{"durationSeconds":8}}
    , out.written());
}

test "gather thoughts, words, a picture and the usage from one answer" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const root = try json.parse(a,
        \\{"candidates":[{"content":{"role":"model","parts":[{"text":"Hmm.","thought":true},{"text":"Here:"},
        \\{"inlineData":{"mimeType":"image/png","data":"iVBORw0KGgo="}}]},"finishReason":"STOP"}],
        \\"usageMetadata":{"promptTokenCount":9,"candidatesTokenCount":3,"thoughtsTokenCount":4},"modelVersion":"gemini-3.1-flash-image"}
    );
    var harvest: Harvest = .{};
    try harvest.gather(a, root);
    try std.testing.expectEqualStrings("Here:", harvest.text.items);
    try std.testing.expectEqualStrings("Hmm.", harvest.reasoning.items);
    try std.testing.expectEqual(@as(usize, 1), harvest.images.items.len);
    try std.testing.expectEqual(Finish.stop, finishOf(harvest.finish_reason));
    try std.testing.expectEqual(@as(?u64, 7), harvest.usage.output_tokens);
}

test "a finished Veo operation, and a filtered one" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var client: Client = undefined;
    client.failure = .{};

    var video: Video = .{ .arena = undefined };
    try parseOperation(&client, try json.parse(a,
        \\{"name":"models/veo/operations/1","done":true,"response":{"@type":"x","generateVideoResponse":
        \\{"generatedSamples":[{"video":{"uri":"https://generativelanguage.googleapis.com/v1beta/files/1:download?alt=media"}}]}}}
    ), &video);
    try std.testing.expectEqual(Video.Status.completed, video.status);
    try std.testing.expect(std.mem.endsWith(u8, video.url.?, "alt=media"));

    video = .{ .arena = undefined };
    try parseOperation(&client, try json.parse(a,
        \\{"name":"models/veo/operations/2","done":true,"response":{"generateVideoResponse":{"raiMediaFilteredCount":1,"raiMediaFilteredReasons":["filtered"]}}}
    ), &video);
    try std.testing.expectEqual(Video.Status.failed, video.status);
    try std.testing.expectEqualStrings("filtered", video.message);
}
