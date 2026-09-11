// SPDX-License-Identifier: BSL-1.0

//! The whole way down, against servers on this machine.
//!
//! Two `std.http.Server`s run beside the tests. The first plays a provider -
//! all three APIs, on the paths each defines - and answers a request whose
//! key or shape is wrong the way a provider would, so a test that sends the
//! wrong thing fails on the status that comes back. The second plays
//! somebody's storage: it serves pictures and videos and refuses, with a
//! 400, any request that arrives carrying a key. Links and redirects lead
//! there, and a key that leaked would fail the test that followed them.
//!
//! No test here touches the network beyond this machine, and none needs a
//! key of any provider's.

const std = @import("std");
const Io = std.Io;
const http = std.http;
const ai = @import("fluxion_ai");

const testing = std.testing;
const gpa = testing.allocator;

const key = "test-key";
const png = "\x89PNG\r\n\x1a\n\x00\x00\x00\x0dIHDR fake picture";
const png_b64: []const u8 = b64: {
    const encoder = std.base64.standard.Encoder;
    var out: [encoder.calcSize(png.len)]u8 = undefined;
    _ = encoder.encode(&out, png);
    const final = out;
    break :b64 &final;
};
const mp4 = "\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00 fake video";

/// What the handler needs of a request, copied out of its head before
/// reading the body invalidates it.
const Seen = struct {
    method: http.Method,
    target: []const u8,
    authorization: ?[]const u8 = null,
    x_api_key: ?[]const u8 = null,
    x_goog_api_key: ?[]const u8 = null,
    anthropic_version: ?[]const u8 = null,
    content_type: ?[]const u8 = null,

    fn credentials(s: Seen) bool {
        return s.authorization != null or s.x_api_key != null or s.x_goog_api_key != null;
    }
};

const Mock = struct {
    server: Io.net.Server,
    port: u16,
    role: enum { provider, storage, silent },
    /// The other server's port, to point links and redirects at.
    storage_port: u16 = 0,
    /// Times the video has been asked after, so it can finish on the second.
    polls: std.atomic.Value(u32) = .init(0),
    stopping: std.atomic.Value(bool) = .init(false),

    fn listen(io: Io, role: @FieldType(Mock, "role")) !Mock {
        const address = try Io.net.IpAddress.parse("127.0.0.1", 0);
        const server = try address.listen(io, .{ .reuse_address = true });
        return .{ .server = server, .port = server.socket.address.getPort(), .role = role };
    }

    fn run(m: *Mock, io: Io) void {
        while (true) {
            const stream = m.server.accept(io) catch return;
            defer stream.close(io);
            if (m.stopping.load(.acquire)) return;
            m.serve(io, stream) catch |err| std.debug.print("mock server: {t}\n", .{err});
        }
    }

    /// Wake the accept loop so that it sees `stopping`, and close up.
    fn stop(m: *Mock, io: Io) void {
        m.stopping.store(true, .release);
        const address = Io.net.IpAddress.parse("127.0.0.1", m.port) catch unreachable;
        if (address.connect(io, .{ .mode = .stream })) |stream| stream.close(io) else |_| {}
    }

    fn serve(m: *Mock, io: Io, stream: Io.net.Stream) !void {
        var receive_buffer: [64 * 1024]u8 = undefined;
        var send_buffer: [16 * 1024]u8 = undefined;
        var reader = stream.reader(io, &receive_buffer);
        var writer = stream.writer(io, &send_buffer);
        var server: http.Server = .init(&reader.interface, &writer.interface);

        var request = server.receiveHead() catch |err| switch (err) {
            error.HttpConnectionClosing => return,
            else => |e| return e,
        };

        var arena: std.heap.ArenaAllocator = .init(gpa);
        defer arena.deinit();
        const a = arena.allocator();

        var seen: Seen = .{ .method = request.head.method, .target = try a.dupe(u8, request.head.target) };
        var it = request.iterateHeaders();
        while (it.next()) |header| {
            const value = try a.dupe(u8, header.value);
            if (std.ascii.eqlIgnoreCase(header.name, "authorization")) seen.authorization = value;
            if (std.ascii.eqlIgnoreCase(header.name, "x-api-key")) seen.x_api_key = value;
            if (std.ascii.eqlIgnoreCase(header.name, "x-goog-api-key")) seen.x_goog_api_key = value;
            if (std.ascii.eqlIgnoreCase(header.name, "anthropic-version")) seen.anthropic_version = value;
            if (std.ascii.eqlIgnoreCase(header.name, "content-type")) seen.content_type = value;
        }
        var body_buffer: [4096]u8 = undefined;
        const body = try request.readerExpectNone(&body_buffer).allocRemaining(a, .limited(8 << 20));

        switch (m.role) {
            .provider => try m.provide(a, &request, seen, body),
            .storage => try store(&request, seen),
            // Reads every byte, so that the close is a clean one, and says
            // nothing.
            .silent => {},
        }
    }

    fn provide(m: *Mock, a: std.mem.Allocator, request: *http.Server.Request, seen: Seen, body: []const u8) !void {
        const target = seen.target;
        const has = struct {
            fn f(haystack: []const u8, needle: []const u8) bool {
                return std.mem.find(u8, haystack, needle) != null;
            }
        }.f;

        // ---- OpenAI's shape ------------------------------------------------
        if (std.mem.startsWith(u8, target, "/v1/") and !std.mem.startsWith(u8, target, "/v1/messages")) {
            if (!std.mem.eql(u8, seen.authorization orelse "", "Bearer " ++ key))
                return json(request, .unauthorized,
                    \\{"error":{"message":"Incorrect API key provided: wrong-key.","type":"invalid_request_error","code":"invalid_api_key"}}
                );
        }
        if (std.mem.eql(u8, target, "/v1/chat/completions")) {
            if (has(body, "\"model\":\"broke\"")) return json(request, .too_many_requests,
                \\{"error":{"message":"You exceeded your current quota.","type":"insufficient_quota","code":"insufficient_quota"}}
            );
            if (has(body, "\"stream\":true")) {
                if (!has(body, "\"stream_options\":{\"include_usage\":true}")) return json(request, .bad_request, "{\"error\":{\"message\":\"no stream_options\"}}");
                return events(request, &.{
                    "data: {\"id\":\"c1\",\"model\":\"mock-1\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}\n\n",
                    ": keep-alive\n\n",
                    "data: {\"id\":\"c1\",\"model\":\"mock-1\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"Think.\"}}]}\n\n",
                    "data: {\"id\":\"c1\",\"model\":\"mock-1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Szia, \"}}]}\n\ndata: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"con",
                    "tent\":\"világ!\"}}]}\r\n\r\n",
                    "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
                    "data: {\"id\":\"c1\",\"choices\":[],\"usage\":{\"prompt_tokens\":11,\"completion_tokens\":3}}\n\n",
                    "data: [DONE]\n\n",
                });
            }
            // The request must say what the test asked for, in OpenAI's words.
            if (!has(body, "\"max_completion_tokens\":50") or !has(body, "\"temperature\":0.25") or
                !has(body, "\"role\":\"system\",\"content\":\"Be brief.\"") or !has(body, "\"seed\":7"))
                return json(request, .bad_request, "{\"error\":{\"message\":\"unexpected body\"}}");
            return json(request, .ok,
                \\{"id":"c2","object":"chat.completion","model":"mock-1","choices":[{"index":0,
                \\"message":{"role":"assistant","content":"Szia!","reasoning_content":"Short."},"finish_reason":"stop"}],
                \\"usage":{"prompt_tokens":20,"completion_tokens":2,"total_tokens":22}}
            );
        }
        if (std.mem.eql(u8, target, "/v1/images/generations")) {
            const link = try std.fmt.allocPrint(a, "http://127.0.0.1:{d}/cdn/picture.png", .{m.storage_port});
            return json(request, .ok, try std.fmt.allocPrint(a,
                \\{{"created":1,"data":[{{"b64_json":"{s}","revised_prompt":"a fox, painted"}},{{"url":"{s}"}}],"usage":{{"input_tokens":5,"output_tokens":100}}}}
            , .{ png_b64, link }));
        }
        if (std.mem.eql(u8, target, "/v1/images/edits")) {
            if (!std.mem.startsWith(u8, seen.content_type orelse "", "multipart/form-data; boundary=") or
                !has(body, "name=\"image\"; filename=\"reference-1.png\"\r\ncontent-type: image/png\r\n\r\n" ++ png) or
                !has(body, "name=\"background\"\r\n\r\ntransparent\r\n"))
                return json(request, .bad_request, "{\"error\":{\"message\":\"unexpected form\"}}");
            return json(request, .ok, "{\"data\":[{\"b64_json\":\"" ++ png_b64 ++ "\"}]}");
        }
        if (std.mem.eql(u8, target, "/v1/videos")) {
            if (!has(body, "name=\"model\"\r\n\r\nsora-2\r\n") or !has(body, "name=\"seconds\"\r\n\r\n4\r\n"))
                return json(request, .bad_request, "{\"error\":{\"message\":\"unexpected form\"}}");
            return json(request, .ok, "{\"id\":\"video_1\",\"object\":\"video\",\"status\":\"queued\",\"progress\":0}");
        }
        if (std.mem.eql(u8, target, "/v1/videos/video_1")) {
            if (m.polls.fetchAdd(1, .monotonic) == 0)
                return json(request, .ok, "{\"id\":\"video_1\",\"status\":\"in_progress\",\"progress\":40}");
            return json(request, .ok, "{\"id\":\"video_1\",\"status\":\"completed\",\"progress\":100}");
        }
        if (std.mem.eql(u8, target, "/v1/videos/video_1/content")) {
            // The provider's own link sends the client on to storage.
            return redirect(request, try std.fmt.allocPrint(a, "http://127.0.0.1:{d}/cdn/video.mp4", .{m.storage_port}));
        }
        if (std.mem.eql(u8, target, "/v1/models")) {
            return json(request, .ok, "{\"object\":\"list\",\"data\":[{\"id\":\"mock-1\"},{\"id\":\"mock-2\"}]}");
        }
        if (std.mem.eql(u8, target, "/v1/embeddings")) {
            return json(request, .ok, "{\"data\":[{\"embedding\":[0.1,0.2]}]}");
        }

        // ---- Anthropic's -----------------------------------------------------
        if (std.mem.eql(u8, target, "/v1/messages")) {
            if (!std.mem.eql(u8, seen.x_api_key orelse "", key) or seen.anthropic_version == null or seen.authorization != null)
                return json(request, .unauthorized, "{\"type\":\"error\",\"error\":{\"type\":\"authentication_error\",\"message\":\"invalid x-api-key\"}}");
            if (!has(body, "\"max_tokens\":4096") or !has(body, "\"system\":\"Be brief.\""))
                return json(request, .bad_request, "{\"type\":\"error\",\"error\":{\"type\":\"invalid_request_error\",\"message\":\"unexpected body\"}}");
            if (has(body, "\"stream\":true")) return events(request, &.{
                "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude-mock\",\"usage\":{\"input_tokens\":9,\"output_tokens\":1}}}\n\n",
                "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}}\n\n",
                "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"Hmm.\"}}\n\n",
                "event: ping\ndata: {\"type\": \"ping\"}\n\n",
                "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\n",
                "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\" Claude\"}}\n\n",
                "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"max_tokens\"},\"usage\":{\"output_tokens\":6}}\n\n",
                "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
            });
            return json(request, .ok,
                \\{"id":"msg_2","type":"message","model":"claude-mock","content":[{"type":"text","text":"Hello"}],"stop_reason":"end_turn","usage":{"input_tokens":9,"output_tokens":1}}
            );
        }

        // ---- Google's ------------------------------------------------------------
        if (std.mem.startsWith(u8, target, "/v1beta/")) {
            if (!std.mem.eql(u8, seen.x_goog_api_key orelse "", key) or seen.authorization != null)
                return json(request, .bad_request, "[{\"error\":{\"code\":400,\"message\":\"API key not valid. Please pass a valid API key.\",\"status\":\"INVALID_ARGUMENT\"," ++
                    "\"details\":[{\"@type\":\"type.googleapis.com/google.rpc.ErrorInfo\",\"reason\":\"API_KEY_INVALID\"}]}}]");
        }
        if (std.mem.eql(u8, target, "/v1beta/models/gemini-mock:streamGenerateContent?alt=sse")) return events(request, &.{
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Plan.\",\"thought\":true}],\"role\":\"model\"}}],\"modelVersion\":\"gemini-mock-001\"}\r\n\r\n",
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Jó \"}],\"role\":\"model\"}}]}\r\n\r\n",
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"napot!\"}],\"role\":\"model\"},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":4,\"candidatesTokenCount\":3,\"thoughtsTokenCount\":2}}\r\n\r\n",
        });
        if (std.mem.eql(u8, target, "/v1beta/models/gemini-mock:generateContent")) {
            if (!has(body, "\"systemInstruction\":{\"parts\":[{\"text\":\"Be brief.\"}]}"))
                return json(request, .bad_request, "{\"error\":{\"message\":\"unexpected body\"}}");
            return json(request, .ok,
                \\{"candidates":[{"content":{"parts":[{"text":"Here is one."},{"inlineData":{"mimeType":"image/png","data":"
            ++ png_b64 ++
                \\"}}],"role":"model"},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":4,"candidatesTokenCount":1290}}
            );
        }
        if (std.mem.eql(u8, target, "/v1beta/models/veo-mock:predictLongRunning")) {
            if (!has(body, "\"parameters\":{\"aspectRatio\":\"16:9\",\"durationSeconds\":8}"))
                return json(request, .bad_request, "{\"error\":{\"message\":\"unexpected body\"}}");
            return json(request, .ok, "{\"name\":\"models/veo-mock/operations/op1\"}");
        }
        if (std.mem.eql(u8, target, "/v1beta/models/veo-mock/operations/op1")) {
            return json(request, .ok, try std.fmt.allocPrint(a,
                \\{{"name":"models/veo-mock/operations/op1","done":true,"response":{{"generateVideoResponse":{{"generatedSamples":[{{"video":{{"uri":"http://127.0.0.1:{d}/v1beta/files/v1:download?alt=media"}}}}]}}}}}}
            , .{m.port}));
        }
        if (std.mem.eql(u8, target, "/v1beta/files/v1:download?alt=media")) {
            return redirect(request, try std.fmt.allocPrint(a, "http://127.0.0.1:{d}/cdn/video.mp4", .{m.storage_port}));
        }

        return json(request, .not_found, "{\"error\":{\"message\":\"no such endpoint on the mock\"}}");
    }

    fn store(request: *http.Server.Request, seen: Seen) !void {
        if (seen.credentials()) return json(request, .bad_request, "{\"error\":\"a key was sent to storage\"}");
        if (std.mem.eql(u8, seen.target, "/cdn/picture.png")) return bytes(request, png, "image/png");
        if (std.mem.eql(u8, seen.target, "/cdn/video.mp4")) return bytes(request, mp4, "video/mp4");
        return json(request, .not_found, "{\"error\":\"not here\"}");
    }
};

fn json(request: *http.Server.Request, status: http.Status, text: []const u8) !void {
    try request.respond(text, .{
        .status = status,
        .keep_alive = false,
        .extra_headers = &.{.{ .name = "content-type", .value = "application/json" }},
    });
}

fn bytes(request: *http.Server.Request, content: []const u8, content_type: []const u8) !void {
    try request.respond(content, .{
        .keep_alive = false,
        .extra_headers = &.{.{ .name = "content-type", .value = content_type }},
    });
}

fn redirect(request: *http.Server.Request, location: []const u8) !void {
    try request.respond("", .{
        .status = .found,
        .keep_alive = false,
        .extra_headers = &.{.{ .name = "location", .value = location }},
    });
}

/// A stream, one flush per piece, so that the client meets events split
/// across reads and several in one.
fn events(request: *http.Server.Request, pieces: []const []const u8) !void {
    var buffer: [4096]u8 = undefined;
    var body = try request.respondStreaming(&buffer, .{ .respond_options = .{
        .keep_alive = false,
        .extra_headers = &.{.{ .name = "content-type", .value = "text/event-stream" }},
    } });
    for (pieces) |piece| {
        try body.writer.writeAll(piece);
        try body.writer.flush(); // into a chunk
        try body.flush(); // and onto the wire
    }
    try body.end();
}

/// Both servers, running, for the length of one test.
const Fixture = struct {
    provider: Mock,
    storage: Mock,
    provider_task: Io.Future(void) = undefined,
    storage_task: Io.Future(void) = undefined,

    fn start(f: *Fixture, io: Io) !void {
        f.storage = try .listen(io, .storage);
        f.provider = try .listen(io, .provider);
        f.provider.storage_port = f.storage.port;
        f.provider_task = try io.concurrent(Mock.run, .{ &f.provider, io });
        f.storage_task = try io.concurrent(Mock.run, .{ &f.storage, io });
    }

    fn finish(f: *Fixture, io: Io) void {
        f.provider.stop(io);
        f.storage.stop(io);
        f.provider_task.await(io);
        f.storage_task.await(io);
        f.provider.server.deinit(io);
        f.storage.server.deinit(io);
    }

    fn base(f: *Fixture, buffer: []u8, path: []const u8) []const u8 {
        return std.fmt.bufPrint(buffer, "http://127.0.0.1:{d}{s}", .{ f.provider.port, path }) catch unreachable;
    }
};

test "OpenAI-shaped: a chat, a stream, the errors" {
    const io = testing.io;
    var fixture: Fixture = undefined;
    try fixture.start(io);
    defer fixture.finish(io);

    var base_buffer: [64]u8 = undefined;
    var provider: ai.Provider = .compatible(fixture.base(&base_buffer, "/v1"), key);
    provider.max_tokens_field = .max_completion_tokens;
    provider.stream_usage = true;
    var client: ai.Client = .init(gpa, io, provider);
    defer client.deinit();

    {
        var answer = try client.chat(.{
            .model = "mock-1",
            .system = "Be brief.",
            .messages = &.{.user("Szia!")},
            .max_tokens = 50,
            .temperature = 0.25,
            .extra = "{\"seed\":7}",
        });
        defer answer.deinit();
        try testing.expectEqualStrings("Szia!", answer.text);
        try testing.expectEqualStrings("Short.", answer.reasoning);
        try testing.expectEqual(ai.Finish.stop, answer.finish);
        try testing.expectEqual(@as(?u64, 20), answer.usage.input_tokens);
        try testing.expect(std.mem.find(u8, answer.raw, "\"total_tokens\":22") != null);
    }
    {
        const stream = try client.stream(.{ .model = "mock-1", .messages = &.{.user("Szia!")} });
        defer stream.deinit();
        var text: std.ArrayList(u8) = .empty;
        defer text.deinit(gpa);
        var reasoning_events: usize = 0;
        while (try stream.next()) |event| switch (event) {
            .text => |t| try text.appendSlice(gpa, t),
            .reasoning => reasoning_events += 1,
            .image => return error.TestUnexpectedResult,
        };
        try testing.expectEqualStrings("Szia, világ!", text.items);
        try testing.expectEqualStrings("Szia, világ!", stream.text.items);
        try testing.expectEqualStrings("Think.", stream.reasoning.items);
        try testing.expectEqual(@as(usize, 1), reasoning_events);
        try testing.expectEqual(ai.Finish.stop, stream.finish);
        try testing.expectEqual(@as(?u64, 3), stream.usage.output_tokens);
        try testing.expectEqualStrings("mock-1", stream.model);
        try testing.expectEqual(null, try stream.next());
    }

    // Out of money is not the same as too fast.
    try testing.expectError(error.OutOfCredit, client.chat(.{ .model = "broke", .messages = &.{.user("hi")} }));
    try testing.expectEqualStrings("You exceeded your current quota.", client.failure.message());
    try testing.expectEqual(@as(u16, 429), client.failure.status);

    // The wrong key, and what the provider said about it.
    client.provider.api_key = "wrong-key";
    try testing.expectError(error.Unauthorized, client.chat(.{ .model = "mock-1", .messages = &.{.user("hi")} }));
    try testing.expectEqualStrings("Incorrect API key provided: wrong-key.", client.failure.message());
    client.provider.api_key = key;

    // `extra` that is not an object is caught before anything is sent.
    try testing.expectError(error.InvalidExtra, client.chat(.{ .model = "mock-1", .messages = &.{.user("hi")}, .extra = "[1]" }));

    var models = try client.listModels();
    defer models.deinit();
    try testing.expectEqual(@as(usize, 2), models.items.len);
    try testing.expectEqualStrings("mock-2", models.items[1].id);

    var response = try client.call(.POST, "/embeddings", "{\"input\":\"x\"}");
    defer response.deinit();
    try testing.expectEqual(http.Status.ok, response.status);
    try testing.expect(std.mem.startsWith(u8, response.body, "{\"data\""));

    try testing.expectError(error.NotFound, client.call(.GET, "/nowhere", null));
}

test "OpenAI-shaped: pictures, fetched without the key, and edits as a form" {
    const io = testing.io;
    var fixture: Fixture = undefined;
    try fixture.start(io);
    defer fixture.finish(io);

    var base_buffer: [64]u8 = undefined;
    var client: ai.Client = .init(gpa, io, .compatible(fixture.base(&base_buffer, "/v1"), key));
    defer client.deinit();

    {
        var result = try client.generateImages(.{ .model = "gpt-image-mock", .prompt = "a fox", .count = 2 });
        defer result.deinit();
        try testing.expectEqual(@as(usize, 2), result.images.len);
        try testing.expectEqualSlices(u8, png, result.images[0].bytes);
        try testing.expectEqualStrings("image/png", result.images[0].mime_type);
        try testing.expectEqualStrings("a fox, painted", result.images[0].revised_prompt.?);
        try testing.expectEqualStrings("png", result.images[0].extension());
        // The second came as a link to storage, and was fetched from there
        // - without the key, or storage would have refused it.
        try testing.expectEqualSlices(u8, png, result.images[1].bytes);
        try testing.expect(result.images[1].url != null);
        try testing.expectEqual(@as(?u64, 100), result.usage.output_tokens);
    }
    {
        var result = try client.generateImages(.{
            .model = "gpt-image-mock",
            .prompt = "make it transparent",
            .references = &.{.fromBytes(png)},
            .extra = "{\"background\":\"transparent\"}",
        });
        defer result.deinit();
        try testing.expectEqual(@as(usize, 1), result.images.len);
    }
}

test "OpenAI-shaped: a video from start to file, through a redirect to storage" {
    const io = testing.io;
    var fixture: Fixture = undefined;
    try fixture.start(io);
    defer fixture.finish(io);

    var base_buffer: [64]u8 = undefined;
    var client: ai.Client = .init(gpa, io, .compatible(fixture.base(&base_buffer, "/v1"), key));
    defer client.deinit();

    var started = try client.startVideo(.{ .model = "sora-2", .prompt = "waves", .seconds = 4 });
    defer started.deinit();
    try testing.expectEqualStrings("video_1", started.id);
    try testing.expectEqual(ai.Video.Status.queued, started.status);

    var video = try client.waitVideo(started.id, .{ .poll_interval = .fromMilliseconds(10) });
    defer video.deinit();
    try testing.expectEqual(ai.Video.Status.completed, video.status);
    try testing.expectEqual(@as(?u8, 100), video.progress);

    var file: Io.Writer.Allocating = .init(gpa);
    defer file.deinit();
    const n = try client.downloadVideo(&video, &file.writer);
    try testing.expectEqual(@as(u64, mp4.len), n);
    try testing.expectEqualSlices(u8, mp4, file.written());
    try testing.expectEqualStrings("video/mp4", ai.media.sniff(file.written()).?);
}

test "Anthropic-shaped: its own key header, and a stream of named events" {
    const io = testing.io;
    var fixture: Fixture = undefined;
    try fixture.start(io);
    defer fixture.finish(io);

    var base_buffer: [64]u8 = undefined;
    var provider: ai.Provider = .anthropic(key);
    provider.base_url = fixture.base(&base_buffer, "/v1");
    var client: ai.Client = .init(gpa, io, provider);
    defer client.deinit();

    {
        var answer = try client.chat(.{ .model = "claude-mock", .system = "Be brief.", .messages = &.{.user("Hi")} });
        defer answer.deinit();
        try testing.expectEqualStrings("Hello", answer.text);
        try testing.expectEqual(ai.Finish.stop, answer.finish);
    }
    {
        const stream = try client.stream(.{ .model = "claude-mock", .system = "Be brief.", .messages = &.{.user("Hi")} });
        defer stream.deinit();
        while (try stream.next()) |_| {}
        try testing.expectEqualStrings("Hello Claude", stream.text.items);
        try testing.expectEqualStrings("Hmm.", stream.reasoning.items);
        try testing.expectEqual(ai.Finish.length, stream.finish);
        try testing.expectEqualStrings("max_tokens", stream.finish_reason);
        try testing.expectEqual(@as(?u64, 9), stream.usage.input_tokens);
        try testing.expectEqual(@as(?u64, 6), stream.usage.output_tokens);
        try testing.expectEqualStrings("msg_1", stream.id);
    }

    try testing.expectError(error.Unsupported, client.generateImages(.{ .model = "x", .prompt = "x" }));
    try testing.expect(client.failure.message().len > 0);
}

test "Gemini-shaped: a stream, a picture in an answer, and a Veo video" {
    const io = testing.io;
    var fixture: Fixture = undefined;
    try fixture.start(io);
    defer fixture.finish(io);

    var base_buffer: [64]u8 = undefined;
    var provider: ai.Provider = .gemini(key);
    provider.base_url = fixture.base(&base_buffer, "/v1beta");
    var client: ai.Client = .init(gpa, io, provider);
    defer client.deinit();

    {
        const stream = try client.stream(.{ .model = "gemini-mock", .messages = &.{.user("Szia")} });
        defer stream.deinit();
        while (try stream.next()) |_| {}
        try testing.expectEqualStrings("Jó napot!", stream.text.items);
        try testing.expectEqualStrings("Plan.", stream.reasoning.items);
        try testing.expectEqual(ai.Finish.stop, stream.finish);
        try testing.expectEqual(@as(?u64, 5), stream.usage.output_tokens);
        try testing.expectEqualStrings("gemini-mock-001", stream.model);
    }
    {
        var answer = try client.chat(.{ .model = "gemini-mock", .system = "Be brief.", .messages = &.{.user("Draw")} });
        defer answer.deinit();
        try testing.expectEqualStrings("Here is one.", answer.text);
        try testing.expectEqual(@as(usize, 1), answer.images.len);
        try testing.expectEqualSlices(u8, png, answer.images[0].bytes);
    }
    {
        var started = try client.startVideo(.{ .model = "veo-mock", .prompt = "waves", .aspect_ratio = "16:9", .seconds = 8 });
        defer started.deinit();
        try testing.expectEqualStrings("models/veo-mock/operations/op1", started.id);
        var video = try client.waitVideo(started.id, .{ .poll_interval = .fromMilliseconds(10) });
        defer video.deinit();
        // The file is on the provider's host, so the key goes with it; the
        // redirect after it is to storage, so the key stays behind.
        const file = try client.downloadAlloc(gpa, video.url.?);
        defer gpa.free(file);
        try testing.expectEqualSlices(u8, mp4, file);
    }

    // Google's errors can come as an array, and a wrong key as a 400; both
    // are read for what they mean.
    client.provider.api_key = "wrong-key";
    try testing.expectError(error.Unauthorized, client.chat(.{ .model = "gemini-mock", .messages = &.{.user("x")} }));
    try testing.expectEqualStrings("API key not valid. Please pass a valid API key.", client.failure.message());
}

test "a server that hangs up without answering" {
    const io = testing.io;
    var silent = try Mock.listen(io, .silent);
    var task = try io.concurrent(Mock.run, .{ &silent, io });
    defer {
        silent.stop(io);
        task.await(io);
        silent.server.deinit(io);
    }

    var base_buffer: [64]u8 = undefined;
    const base = try std.fmt.bufPrint(&base_buffer, "http://127.0.0.1:{d}/v1", .{silent.port});
    var client: ai.Client = .init(gpa, io, .compatible(base, null));
    defer client.deinit();

    // A network failure comes through as the network's own error, and
    // `failure` says what was being attempted when it happened.
    try testing.expectError(error.HttpConnectionClosing, client.chat(.{ .model = "x", .messages = &.{.user("x")} }));
    var expected_buffer: [128]u8 = undefined;
    const expected = try std.fmt.bufPrint(&expected_buffer, "receiving the answer from http://127.0.0.1:{d}/v1/chat/completions: HttpConnectionClosing", .{silent.port});
    try testing.expectEqualStrings(expected, client.failure.message());
}
