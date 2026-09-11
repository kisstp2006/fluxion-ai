// SPDX-License-Identifier: BSL-1.0

//! The wire: one HTTP exchange with a provider, and everything that can go
//! wrong with one.
//!
//! This is `std.http.Client` with three things added. The key goes in the
//! header the provider wants it in, and only ever to the provider's own
//! host: a picture's link points at somebody's storage, and a redirect can
//! point anywhere, so both are followed without it. An answer that is not a
//! success is read for the provider's explanation, which is kept in
//! `Client.failure`, and becomes an error that says which kind of failure it
//! was. And a request can be a JSON body or a multipart form, which is what
//! OpenAI's picture edits and videos take.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Io = std.Io;
const http = std.http;
const Uri = std.Uri;

const Client = @import("Client.zig");
const Provider = @import("Provider.zig");
const json = @import("json.zig");

/// What went wrong, in the words of whoever said it.
///
/// A Zig error cannot carry a message, and a provider's message is usually
/// the part worth reading: "This model's maximum context length is 128000
/// tokens", "Your credit balance is too low". Every call on a `Client` clears
/// `Client.failure` when it starts and fills it when it fails.
pub const Failure = struct {
    /// The HTTP status of the answer that failed, or 0 when it was not an
    /// answer that failed - a stream cut short, a video that did not render.
    status: u16 = 0,
    buffer: [2048]u8 = undefined,
    len: usize = 0,

    /// The explanation, cut to fit if it was long. Empty when nothing failed.
    pub fn message(f: *const Failure) []const u8 {
        return f.buffer[0..f.len];
    }

    pub fn clear(f: *Failure) void {
        f.status = 0;
        f.len = 0;
    }

    pub fn set(f: *Failure, status: u16, text: []const u8) void {
        f.status = status;
        f.len = 0;
        f.append(std.mem.trim(u8, text, " \t\r\n"));
    }

    pub fn print(f: *Failure, status: u16, comptime fmt: []const u8, args: anytype) void {
        f.status = status;
        var w: Io.Writer = .fixed(&f.buffer);
        w.print(fmt, args) catch {};
        f.len = w.end;
        f.len = utf8Floor(f.buffer[0..f.len]);
    }

    fn append(f: *Failure, text: []const u8) void {
        const n = @min(text.len, f.buffer.len - f.len);
        @memcpy(f.buffer[f.len..][0..n], text[0..n]);
        f.len += n;
        f.len = utf8Floor(f.buffer[0..f.len]);
    }

    /// The length of `bytes` without a character cut in half at the end.
    fn utf8Floor(bytes: []const u8) usize {
        var end = bytes.len;
        var back: usize = 0;
        while (end > 0 and back < 4 and bytes[end - 1] & 0xc0 == 0x80) : (back += 1) end -= 1;
        if (end > 0 and bytes[end - 1] >= 0xc0) {
            const want = std.unicode.utf8ByteSequenceLength(bytes[end - 1]) catch 1;
            if (want > back + 1) return end - 1;
        }
        return bytes.len;
    }
};

/// The ways a call to a provider fails that are about the provider rather
/// than the network. The network's own - `error.ConnectionRefused` when
/// Ollama is not running, `error.TlsInitializationFailed` - come through as
/// `std.http.Client` returns them.
pub const Error = error{
    /// 400, 409, 413, 422: the provider could not use what was sent - a
    /// field it does not know, a picture it will not take, a prompt too long.
    BadRequest,
    /// 401: no key, or not one this provider recognises.
    Unauthorized,
    /// 402, or a 429 that says the quota is spent: the account is out of
    /// credit, and waiting will not help.
    OutOfCredit,
    /// 403: a key it recognises, for something it will not do.
    Forbidden,
    /// 404: no such model, or no such endpoint on this provider.
    NotFound,
    /// 429: too many requests; worth trying again later.
    RateLimited,
    /// 5xx, and Anthropic's 529: the provider's trouble.
    ServerError,
    /// A status that is none of the above and not a success.
    UnexpectedStatus,
    /// An answer arrived, but not in the shape the provider's API promises.
    InvalidResponse,
    /// The answer was longer than `Client.max_response_len`.
    ResponseTooLarge,
    /// Reading the answer failed part way; `Client.failure` says how.
    ReadFailed,
    /// Sending the request failed part way; `Client.failure` says how.
    WriteFailed,
    /// This provider's API has no such thing. Claude does not draw.
    Unsupported,
    /// The provider says the generation itself failed - a video that did not
    /// render, an answer stopped by an error mid-stream.
    GenerationFailed,
    /// `Client.waitVideo` ran out of time.
    Timeout,
    TooManyRedirects,
    /// A URL that does not parse, or a redirect to one.
    InvalidUrl,
    /// A request's `extra` is not a JSON object.
    InvalidExtra,
};

/// The error for a status that is not a success. `body_root` is what came
/// with it, read for the two cases the status alone does not tell apart:
/// OpenAI's 429 that means the money has run out rather than the patience,
/// and Google's 400 that means the key is wrong, which everyone else calls
/// a 401.
pub fn statusError(status: http.Status, body_root: ?std.json.Value) Error {
    return switch (@intFromEnum(status)) {
        400 => {
            const root = if (body_root) |r| switch (r) {
                .array => |a| if (a.items.len > 0) a.items[0] else r,
                else => r,
            } else return error.BadRequest;
            for (json.array(json.at(root, .{ "error", "details" }))) |detail| {
                const reason = json.string(json.at(detail, .{"reason"})) orelse continue;
                if (std.mem.eql(u8, reason, "API_KEY_INVALID")) return error.Unauthorized;
            }
            return error.BadRequest;
        },
        409, 413, 415, 422 => error.BadRequest,
        401 => error.Unauthorized,
        402 => error.OutOfCredit,
        403 => error.Forbidden,
        404 => error.NotFound,
        429 => {
            const code = json.string(json.at(body_root, .{ "error", "code" })) orelse
                json.string(json.at(body_root, .{ "error", "type" })) orelse "";
            if (std.mem.eql(u8, code, "insufficient_quota")) return error.OutOfCredit;
            return error.RateLimited;
        },
        500...599 => error.ServerError,
        else => error.UnexpectedStatus,
    };
}

pub const Payload = union(enum) {
    none,
    json: []const u8,
    form: struct {
        /// `multipart/form-data; boundary=...`
        content_type: []const u8,
        bytes: []const u8,
    },
};

pub const Options = struct {
    method: http.Method = .GET,
    url: []const u8,
    payload: Payload = .none,
    /// Send the provider's key and headers - to the provider's own host,
    /// never anywhere else, whatever this says.
    credentials: bool = true,
    /// What is expected back.
    accept: []const u8 = "application/json",
    /// Ask for the body uncompressed, so that a stream arrives as it is
    /// written rather than whenever a compressor decides to let go of it.
    identity: bool = false,
};

const max_redirects = 5;

/// One request and its response, in flight. Lives where it was opened and
/// must not move: the response points back at the request inside it.
pub const Exchange = struct {
    client: *Client,
    /// Small things that live as long as the exchange: the URL, the headers.
    arena: std.heap.ArenaAllocator,
    request: http.Client.Request = undefined,
    has_request: bool = false,
    response: http.Client.Response = undefined,
    /// The body, decompressed. Valid after `open` returns.
    reader: *Io.Reader = undefined,
    decompress: http.Decompress = undefined,
    status: http.Status = .ok,
    content_length: ?u64 = null,
    /// Copied out of the head before reading the body invalidates it.
    content_type_buffer: [128]u8 = undefined,
    content_type_len: usize = 0,

    /// Send the request and receive the head. A success leaves `reader` at
    /// the start of the body. Anything else is read for the provider's
    /// explanation, recorded in `client.failure`, cleaned up, and returned
    /// as an error; `ex` is then as if it was never opened.
    pub fn open(ex: *Exchange, client: *Client, options: Options) !void {
        ex.* = .{ .client = client, .arena = .init(client.gpa) };
        errdefer ex.deinit();
        const arena = ex.arena.allocator();

        const base_uri = Uri.parse(client.provider.base_url) catch return client.fail(0, error.InvalidUrl, "the provider's base URL does not parse: {s}", .{client.provider.base_url});

        var url: []const u8 = try arena.dupe(u8, options.url);
        var method = options.method;
        var payload = options.payload;
        var redirects: usize = 0;
        while (true) {
            const uri = Uri.parse(url) catch return client.fail(0, error.InvalidUrl, "not a URL: {s}", .{url});
            const trusted = options.credentials and sameOrigin(uri, base_uri);

            var headers: std.ArrayList(http.Header) = .empty;
            var authorization: http.Client.Request.Headers.Value = .omit;
            try headers.append(arena, .{ .name = "accept", .value = options.accept });
            if (trusted) {
                const provider = &client.provider;
                if (provider.api_key) |key| switch (provider.auth) {
                    .bearer => authorization = .{ .override = try std.fmt.allocPrint(arena, "Bearer {s}", .{key}) },
                    .header => |name| try headers.append(arena, .{ .name = name, .value = key }),
                    .none => {},
                };
                if (provider.api == .anthropic) {
                    try headers.append(arena, .{ .name = "anthropic-version", .value = provider.anthropic_version });
                }
                try headers.appendSlice(arena, provider.extra_headers);
            }

            ex.request = client.http.request(method, uri, .{
                .redirect_behavior = .unhandled,
                .headers = .{
                    .user_agent = .{ .override = client.user_agent },
                    .authorization = authorization,
                    .accept_encoding = if (options.identity) .omit else .default,
                    .content_type = switch (payload) {
                        .none => .omit,
                        .json => .{ .override = "application/json" },
                        .form => |form| .{ .override = form.content_type },
                    },
                },
                .extra_headers = headers.items,
            }) catch |err| return client.failError(err, "connecting to {s}", .{url});
            ex.has_request = true;

            switch (payload) {
                // A POST with nothing to say still says so, in a length of 0.
                .none => if (method.requestHasBody())
                    try ex.sendBody("")
                else
                    ex.request.sendBodiless() catch return ex.writeFailure(),
                .json => |bytes| try ex.sendBody(bytes),
                .form => |form| try ex.sendBody(form.bytes),
            }

            ex.response = ex.request.receiveHead(&.{}) catch |err| switch (err) {
                error.ReadFailed => return ex.readFailure("receiving the answer"),
                error.WriteFailed => return ex.writeFailure(),
                else => |e| return client.failError(e, "receiving the answer from {s}", .{url}),
            };
            const head = &ex.response.head;
            ex.status = head.status;

            if (head.status.class() == .redirect and head.status != .not_modified) {
                const location = head.location orelse
                    return client.fail(@intFromEnum(head.status), error.InvalidResponse, "a redirect with nowhere to go", .{});
                if (redirects == max_redirects)
                    return client.fail(@intFromEnum(head.status), error.TooManyRedirects, "more than {d} redirects", .{max_redirects});
                // A body cannot be sent twice from here, so only a redirect
                // that asks for a GET is followed for a request that had one.
                if (payload != .none and head.status != .see_other)
                    return client.fail(@intFromEnum(head.status), error.UnexpectedStatus, "the request was redirected to {s}", .{location});
                const next = resolve(arena, uri, location) catch
                    return client.fail(@intFromEnum(head.status), error.InvalidUrl, "a redirect to something that is not a URL: {s}", .{location});
                ex.request.deinit();
                ex.has_request = false;
                if (head.status == .see_other) {
                    method = .GET;
                    payload = .none;
                }
                url = next;
                redirects += 1;
                continue;
            }
            break;
        }

        const head = &ex.response.head;
        ex.content_length = head.content_length;
        if (head.content_type) |content_type| {
            const n = @min(content_type.len, ex.content_type_buffer.len);
            @memcpy(ex.content_type_buffer[0..n], content_type[0..n]);
            ex.content_type_len = n;
        }

        const decompress_buffer: []u8 = switch (head.content_encoding) {
            .identity => &.{},
            .deflate, .gzip => try arena.alloc(u8, std.compress.flate.max_window_len),
            .zstd => try arena.alloc(u8, std.compress.zstd.default_window_len),
            .compress => return client.fail(@intFromEnum(ex.status), error.InvalidResponse, "an answer compressed with `compress`", .{}),
        };
        const transfer_buffer = try arena.alloc(u8, 16 * 1024);
        ex.reader = ex.response.readerDecompressing(transfer_buffer, &ex.decompress, decompress_buffer);

        if (ex.status.class() != .success) {
            const status = ex.status;
            // Read what it said, but not forever: an error body is a
            // sentence, and one that is not is not worth waiting for.
            var scratch: std.heap.ArenaAllocator = .init(client.gpa);
            defer scratch.deinit();
            const body = ex.reader.allocRemaining(scratch.allocator(), .limited(64 * 1024)) catch "";
            const root: ?std.json.Value = json.parse(scratch.allocator(), body) catch null;
            const text = if (root) |r| json.errorMessage(r) orelse body else body;
            if (text.len > 0) {
                client.failure.set(@intFromEnum(status), text);
            } else {
                client.failure.print(@intFromEnum(status), "{d} {s}", .{ @intFromEnum(status), status.phrase() orelse "" });
            }
            return statusError(status, root);
        }
    }

    pub fn deinit(ex: *Exchange) void {
        if (ex.has_request) ex.request.deinit();
        ex.arena.deinit();
        ex.* = undefined;
    }

    pub fn contentType(ex: *const Exchange) ?[]const u8 {
        if (ex.content_type_len == 0) return null;
        return ex.content_type_buffer[0..ex.content_type_len];
    }

    /// The whole body, in memory from `gpa`.
    pub fn readAll(ex: *Exchange, gpa: Allocator) ![]u8 {
        return ex.reader.allocRemaining(gpa, .limited(ex.client.max_response_len)) catch |err| switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            error.StreamTooLong => ex.client.fail(@intFromEnum(ex.status), error.ResponseTooLarge, "the answer is larger than max_response_len ({d} bytes)", .{ex.client.max_response_len}),
            error.ReadFailed => ex.readFailure("reading the answer"),
        };
    }

    /// The whole body, into `w`, however long it is. Returns its length.
    pub fn streamAll(ex: *Exchange, w: *Io.Writer) !u64 {
        return ex.reader.streamRemaining(w) catch |err| switch (err) {
            error.ReadFailed => ex.readFailure("reading the answer"),
            error.WriteFailed => error.WriteFailed,
        };
    }

    fn sendBody(ex: *Exchange, bytes: []const u8) !void {
        ex.request.transfer_encoding = .{ .content_length = bytes.len };
        var body = ex.request.sendBodyUnflushed(&.{}) catch return ex.writeFailure();
        body.writer.writeAll(bytes) catch return ex.writeFailure();
        body.end() catch return ex.writeFailure();
        ex.request.connection.?.flush() catch return ex.writeFailure();
    }

    /// `error.ReadFailed`, with what actually failed in `client.failure`.
    pub fn readFailure(ex: *Exchange, doing: []const u8) Error {
        const client = ex.client;
        if (ex.has_request) {
            if (ex.request.reader.body_err) |err| return client.fail(0, error.ReadFailed, "{s}: {t}", .{ doing, err });
            if (ex.request.connection) |connection| {
                if (connection.getReadError()) |err| return client.fail(0, error.ReadFailed, "{s}: {t}", .{ doing, err });
            }
        }
        return client.fail(0, error.ReadFailed, "{s}: the connection failed", .{doing});
    }

    fn writeFailure(ex: *Exchange) Error {
        const client = ex.client;
        if (ex.has_request) {
            if (ex.request.connection) |connection| {
                if (connection.stream_writer.err) |err| return client.fail(0, error.WriteFailed, "sending the request: {t}", .{err});
            }
        }
        return client.fail(0, error.WriteFailed, "sending the request failed", .{});
    }
};

// ---------------------------------------------------------------------------
// URLs
// ---------------------------------------------------------------------------

/// `base` and `path` joined by exactly one slash. A `path` that is already
/// a whole URL is returned as it is.
pub fn join(arena: Allocator, base: []const u8, path: []const u8) Allocator.Error![]u8 {
    if (std.mem.startsWith(u8, path, "https://") or std.mem.startsWith(u8, path, "http://")) return arena.dupe(u8, path);
    const b = std.mem.trimEnd(u8, base, "/");
    const p = std.mem.trimStart(u8, path, "/");
    return std.fmt.allocPrint(arena, "{s}/{s}", .{ b, p });
}

/// Same scheme, same host, same port: the only place a key is sent.
pub fn sameOrigin(a: Uri, b: Uri) bool {
    if (!std.ascii.eqlIgnoreCase(a.scheme, b.scheme)) return false;
    if (effectivePort(a) != effectivePort(b)) return false;
    var a_buffer: [Io.net.HostName.max_len]u8 = undefined;
    var b_buffer: [Io.net.HostName.max_len]u8 = undefined;
    const a_host = a.getHost(&a_buffer) catch return false;
    const b_host = b.getHost(&b_buffer) catch return false;
    return std.ascii.eqlIgnoreCase(a_host.bytes, b_host.bytes);
}

fn effectivePort(uri: Uri) ?u16 {
    if (uri.port) |port| return port;
    if (std.ascii.eqlIgnoreCase(uri.scheme, "https")) return 443;
    if (std.ascii.eqlIgnoreCase(uri.scheme, "http")) return 80;
    return null;
}

/// Where a `Location` header points, as a whole URL.
fn resolve(arena: Allocator, base: Uri, location: []const u8) ![]u8 {
    var buffer = try arena.alloc(u8, location.len * 2 + 8 * 1024);
    @memcpy(buffer[0..location.len], location);
    var aux = buffer;
    const target = try base.resolveInPlace(location.len, &aux);
    return std.fmt.allocPrint(arena, "{f}", .{target.fmt(.all)});
}

// ---------------------------------------------------------------------------
// Multipart forms
// ---------------------------------------------------------------------------

/// A `multipart/form-data` body: what OpenAI's picture edits and videos take
/// when a file goes with them.
pub const Form = struct {
    out: Io.Writer.Allocating,
    boundary: [boundary_prefix.len + 32]u8,
    content_type_buffer: [64 + boundary_prefix.len + 32]u8 = undefined,

    const boundary_prefix = "fluxion-ai-";

    pub fn init(gpa: Allocator, io: Io) Form {
        var random: [16]u8 = undefined;
        io.random(&random);
        var form: Form = .{ .out = .init(gpa), .boundary = undefined };
        const hex = std.fmt.bytesToHex(random, .lower);
        @memcpy(form.boundary[0..boundary_prefix.len], boundary_prefix);
        @memcpy(form.boundary[boundary_prefix.len..], &hex);
        return form;
    }

    pub fn deinit(form: *Form) void {
        form.out.deinit();
    }

    pub fn field(form: *Form, name: []const u8, value: []const u8) Io.Writer.Error!void {
        const w = &form.out.writer;
        try w.print("--{s}\r\ncontent-disposition: form-data; name=\"{s}\"\r\n\r\n", .{ &form.boundary, name });
        try w.writeAll(value);
        try w.writeAll("\r\n");
    }

    pub fn file(form: *Form, name: []const u8, filename: []const u8, mime_type: []const u8, bytes: []const u8) Io.Writer.Error!void {
        const w = &form.out.writer;
        try w.print("--{s}\r\ncontent-disposition: form-data; name=\"{s}\"; filename=\"{s}\"\r\ncontent-type: {s}\r\n\r\n", .{ &form.boundary, name, filename, mime_type });
        try w.writeAll(bytes);
        try w.writeAll("\r\n");
    }

    /// Every member of `extra` as a field of its own: strings as they are,
    /// anything else as its JSON.
    pub fn fields(form: *Form, extra: ?std.json.ObjectMap) Io.Writer.Error!void {
        const members = extra orelse return;
        var it = members.iterator();
        while (it.next()) |entry| {
            switch (entry.value_ptr.*) {
                .string => |s| try form.field(entry.key_ptr.*, s),
                .number_string => |s| try form.field(entry.key_ptr.*, s),
                else => |v| {
                    var buffer: [4096]u8 = undefined;
                    var w: Io.Writer = .fixed(&buffer);
                    std.json.Stringify.value(v, .{}, &w) catch continue;
                    try form.field(entry.key_ptr.*, w.buffered());
                },
            }
        }
    }

    /// The closing boundary. Returns the body.
    pub fn finish(form: *Form) Io.Writer.Error![]const u8 {
        try form.out.writer.print("--{s}--\r\n", .{&form.boundary});
        return form.out.written();
    }

    pub fn contentType(form: *Form) []const u8 {
        return std.fmt.bufPrint(&form.content_type_buffer, "multipart/form-data; boundary={s}", .{&form.boundary}) catch unreachable;
    }
};

test join {
    const gpa = std.testing.allocator;
    const a = try join(gpa, "https://api.openai.com/v1/", "/chat/completions");
    defer gpa.free(a);
    try std.testing.expectEqualStrings("https://api.openai.com/v1/chat/completions", a);
    const b = try join(gpa, "https://api.deepseek.com", "models");
    defer gpa.free(b);
    try std.testing.expectEqualStrings("https://api.deepseek.com/models", b);
    const c = try join(gpa, "https://x/v1", "https://cdn.example.com/a.png");
    defer gpa.free(c);
    try std.testing.expectEqualStrings("https://cdn.example.com/a.png", c);
}

test sameOrigin {
    const base = try Uri.parse("https://generativelanguage.googleapis.com/v1beta");
    try std.testing.expect(sameOrigin(try Uri.parse("https://GenerativeLanguage.googleapis.com:443/v1beta/files/x:download?alt=media"), base));
    try std.testing.expect(!sameOrigin(try Uri.parse("https://storage.googleapis.com/x.mp4"), base));
    try std.testing.expect(!sameOrigin(try Uri.parse("http://generativelanguage.googleapis.com/v1beta"), base));
    try std.testing.expect(!sameOrigin(try Uri.parse("https://generativelanguage.googleapis.com.evil.example/"), base));
    const local = try Uri.parse("http://127.0.0.1:8080/v1");
    try std.testing.expect(!sameOrigin(try Uri.parse("http://localhost:8080/v1"), local));
    try std.testing.expect(!sameOrigin(try Uri.parse("http://127.0.0.1:8081/v1"), local));
}

test resolve {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const base = try Uri.parse("https://api.openai.com/v1/videos/video_1/content");
    try std.testing.expectEqualStrings("https://cdn.example.com/v.mp4?sig=1", try resolve(a, base, "https://cdn.example.com/v.mp4?sig=1"));
    try std.testing.expectEqualStrings("https://api.openai.com/files/v.mp4", try resolve(a, base, "/files/v.mp4"));
    try std.testing.expectEqualStrings("https://api.openai.com/v1/videos/video_1/other", try resolve(a, base, "other"));
}

test Failure {
    var f: Failure = .{};
    f.set(429, "  slow down \n");
    try std.testing.expectEqualStrings("slow down", f.message());
    try std.testing.expectEqual(@as(u16, 429), f.status);

    // A long message is cut, and never through the middle of a character.
    const long = "é" ** 2000;
    f.set(400, long);
    try std.testing.expect(f.message().len <= f.buffer.len);
    try std.testing.expect(std.unicode.utf8ValidateSlice(f.message()));

    f.print(0, "{s} {d}", .{ "count", 3 });
    try std.testing.expectEqualStrings("count 3", f.message());
}

test Form {
    const gpa = std.testing.allocator;
    var form: Form = .init(gpa, std.testing.io);
    defer form.deinit();
    try form.field("model", "sora-2");
    try form.file("input_reference", "first.png", "image/png", "\x89PNG");
    const body = try form.finish();
    const boundary = &form.boundary;
    try std.testing.expect(std.mem.startsWith(u8, body, "--fluxion-ai-"));
    try std.testing.expect(std.mem.find(u8, body, "name=\"model\"\r\n\r\nsora-2\r\n") != null);
    try std.testing.expect(std.mem.find(u8, body, "filename=\"first.png\"\r\ncontent-type: image/png\r\n\r\n\x89PNG\r\n") != null);
    try std.testing.expect(std.mem.endsWith(u8, body, "--\r\n"));
    try std.testing.expect(std.mem.find(u8, form.contentType(), boundary) != null);
}

test statusError {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const google = try json.parse(a,
        \\{"error":{"code":400,"message":"API key not valid. Please pass a valid API key.","status":"INVALID_ARGUMENT",
        \\"details":[{"@type":"type.googleapis.com/google.rpc.ErrorInfo","reason":"API_KEY_INVALID","domain":"googleapis.com"}]}}
    );
    try std.testing.expectEqual(error.Unauthorized, statusError(.bad_request, google));
    try std.testing.expectEqual(error.BadRequest, statusError(.bad_request, null));
    const quota = try json.parse(a, "{\"error\":{\"message\":\"x\",\"type\":\"insufficient_quota\",\"code\":\"insufficient_quota\"}}");
    try std.testing.expectEqual(error.OutOfCredit, statusError(.too_many_requests, quota));
    try std.testing.expectEqual(error.RateLimited, statusError(.too_many_requests, null));
    try std.testing.expectEqual(error.ServerError, statusError(@enumFromInt(529), null));
    try std.testing.expectEqual(error.OutOfCredit, statusError(.payment_required, null));
}
