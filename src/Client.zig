// SPDX-License-Identifier: BSL-1.0

//! One provider, one connection pool, and every call this library makes.
//!
//! ```zig
//! var client: ai.Client = .init(gpa, io, .deepseek(key));
//! defer client.deinit();
//!
//! var answer = try client.chat(.{
//!     .model = "deepseek-flash",
//!     .messages = &.{.user("Why is the sky blue?")},
//! });
//! defer answer.deinit();
//! ```
//!
//! `provider` is a plain field: set it between calls to ask someone else,
//! over the same connections. A call that fails leaves the provider's own
//! explanation in `failure`. A client can be shared between threads as far
//! as the connections go, but `failure` is whichever call failed last, so a
//! program that wants every message gives each thread a client of its own.
//! Once it has made a call a client must stay where it is: the connections
//! it keeps point back at it.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Io = std.Io;
const fluxion_json = @import("fluxion_json");
const Value = fluxion_json.Value;

const Provider = @import("Provider.zig");
const ChatStream = @import("ChatStream.zig");
const chat_types = @import("chat.zig");
const image_types = @import("image.zig");
const video_types = @import("video.zig");
const json = @import("json.zig");
const media = @import("media.zig");
const transport = @import("transport.zig");
const openai = @import("api/openai.zig");
const anthropic = @import("api/anthropic.zig");
const gemini = @import("api/gemini.zig");

const Client = @This();

pub const version = "0.1.0";

gpa: Allocator,
io: Io,
/// The HTTP client every call goes through. Public, so that a program can
/// set proxies on it (`http.initDefaultProxies`) or bigger buffers.
http: std.http.Client,
/// Who is asked.
provider: Provider,
/// What the last call that failed was told.
failure: transport.Failure = .{},
/// The largest answer read into memory. Downloads into a writer have no
/// limit; everything else - JSON with pictures in it, a line of a stream -
/// is refused past this.
max_response_len: usize = 64 << 20,
user_agent: []const u8 = "fluxion-ai/" ++ version,

pub fn init(gpa: Allocator, io: Io, provider: Provider) Client {
    return .{
        .gpa = gpa,
        .io = io,
        .http = .{ .allocator = gpa, .io = io },
        .provider = provider,
    };
}

pub fn deinit(c: *Client) void {
    c.http.deinit();
    c.* = undefined;
}

// ---------------------------------------------------------------------------
// Words
// ---------------------------------------------------------------------------

/// Ask, and wait for the whole answer.
pub fn chat(c: *Client, request: chat_types.ChatRequest) !chat_types.Chat {
    c.failure.clear();
    var result: chat_types.Chat = .{ .arena = .init(c.gpa) };
    errdefer result.arena.deinit();
    const a = result.arena.allocator();
    switch (c.provider.api) {
        .openai => try openai.chat(c, a, request, &result),
        .anthropic => try anthropic.chat(c, a, request, &result),
        .gemini => try gemini.chat(c, a, request, &result),
    }
    return result;
}

/// Ask, and take the answer as it is written. See `ChatStream`.
pub fn stream(c: *Client, request: chat_types.ChatRequest) !*ChatStream {
    c.failure.clear();
    return ChatStream.open(c, request);
}

// ---------------------------------------------------------------------------
// Pictures
// ---------------------------------------------------------------------------

/// Draw. What comes back is files - see `image`.
pub fn generateImages(c: *Client, request: image_types.ImageRequest) !image_types.Images {
    c.failure.clear();
    var result: image_types.Images = .{ .arena = .init(c.gpa) };
    errdefer result.arena.deinit();
    const a = result.arena.allocator();
    switch (c.provider.api) {
        .openai => try openai.images(c, a, request, &result),
        .gemini => try gemini.images(c, a, request, &result),
        .anthropic => return c.fail(0, error.Unsupported, "Anthropic's API does not generate pictures", .{}),
    }

    if (request.fetch_urls) {
        const images = try a.dupe(image_types.GeneratedImage, result.images);
        for (images) |*image| {
            if (image.bytes.len > 0) continue;
            const url = image.url orelse continue;
            var ex: transport.Exchange = undefined;
            try ex.open(c, .{ .url = url, .accept = "image/*" });
            defer ex.deinit();
            image.bytes = try ex.readAll(a);
            image.mime_type = media.sniff(image.bytes) orelse
                (if (ex.contentType()) |t| try a.dupe(u8, t) else "application/octet-stream");
        }
        result.images = images;
    }
    return result;
}

// ---------------------------------------------------------------------------
// Video
// ---------------------------------------------------------------------------

/// Ask for a video, and return as soon as the provider has taken the job.
/// See `video` for what comes after.
pub fn startVideo(c: *Client, request: video_types.VideoRequest) !video_types.Video {
    c.failure.clear();
    var result: video_types.Video = .{ .arena = .init(c.gpa) };
    errdefer result.arena.deinit();
    const a = result.arena.allocator();
    switch (c.provider.api) {
        .openai => try openai.startVideo(c, a, request, &result),
        .gemini => try gemini.startVideo(c, a, request, &result),
        .anthropic => return c.fail(0, error.Unsupported, "Anthropic's API does not generate video", .{}),
    }
    return result;
}

/// How the video `id` is coming along. A video that failed is an answer
/// here, not an error: `status` is `.failed` and `message` says why.
pub fn videoStatus(c: *Client, id: []const u8) !video_types.Video {
    c.failure.clear();
    var result: video_types.Video = .{ .arena = .init(c.gpa) };
    errdefer result.arena.deinit();
    const a = result.arena.allocator();
    switch (c.provider.api) {
        .openai => try openai.videoStatus(c, a, id, &result),
        .gemini => try gemini.videoStatus(c, a, id, &result),
        .anthropic => return c.fail(0, error.Unsupported, "Anthropic's API does not generate video", .{}),
    }
    return result;
}

/// Ask after the video `id` until it is done. A video that failed is
/// `error.GenerationFailed` here, with the reason in `failure`.
pub fn waitVideo(c: *Client, id: []const u8, options: video_types.WaitOptions) !video_types.Video {
    const start = Io.Clock.awake.now(c.io);
    while (true) {
        var video = try c.videoStatus(id);
        switch (video.status) {
            .completed => return video,
            .failed => {
                c.failure.set(0, if (video.message.len > 0) video.message else "the video failed");
                video.deinit();
                return error.GenerationFailed;
            },
            .queued, .in_progress => video.deinit(),
        }
        if (options.timeout) |timeout| {
            if (start.untilNow(c.io, .awake).nanoseconds >= timeout.nanoseconds) {
                return c.fail(0, error.Timeout, "the video was not done after {d} seconds", .{timeout.toSeconds()});
            }
        }
        try c.io.sleep(options.poll_interval, .awake);
    }
}

/// The finished video's file, into `w`. Returns its length in bytes.
pub fn downloadVideo(c: *Client, video: *const video_types.Video, w: *Io.Writer) !u64 {
    const url = video.url orelse return c.fail(0, error.InvalidUrl, "the video has no file yet (status: {t})", .{video.status});
    return c.download(url, w);
}

// ---------------------------------------------------------------------------
// Everything else
// ---------------------------------------------------------------------------

pub const Model = struct {
    /// What a request names it by.
    id: []const u8,
    /// What it is called, where the provider says (Anthropic, Gemini);
    /// the id otherwise.
    name: []const u8,
};

pub const Models = struct {
    arena: std.heap.ArenaAllocator,
    items: []const Model = &.{},
    raw: []const u8 = "",

    pub fn deinit(models: *Models) void {
        models.arena.deinit();
        models.* = undefined;
    }
};

/// The models this key can use.
pub fn listModels(c: *Client) !Models {
    c.failure.clear();
    var result: Models = .{ .arena = .init(c.gpa) };
    errdefer result.arena.deinit();
    const a = result.arena.allocator();
    switch (c.provider.api) {
        .openai => try openai.models(c, a, &result),
        .anthropic => try anthropic.models(c, a, &result),
        .gemini => try gemini.models(c, a, &result),
    }
    return result;
}

/// Fetch `url` into `w`: a video, a picture's link. The provider's key goes
/// with it only when `url` is on the provider's own host. Returns the length.
pub fn download(c: *Client, url: []const u8, w: *Io.Writer) !u64 {
    c.failure.clear();
    var ex: transport.Exchange = undefined;
    try ex.open(c, .{ .url = url, .accept = "*/*" });
    defer ex.deinit();
    return ex.streamAll(w);
}

/// Fetch `url` into memory from `gpa`, up to `max_response_len`.
pub fn downloadAlloc(c: *Client, gpa: Allocator, url: []const u8) ![]u8 {
    c.failure.clear();
    var ex: transport.Exchange = undefined;
    try ex.open(c, .{ .url = url, .accept = "*/*" });
    defer ex.deinit();
    return ex.readAll(gpa);
}

/// A raw answer: its status, and its body in memory from the client's
/// allocator.
pub const Response = struct {
    gpa: Allocator,
    status: std.http.Status,
    body: []u8,

    pub fn deinit(r: *Response) void {
        r.gpa.free(r.body);
        r.* = undefined;
    }
};

/// Any endpoint this library has no function for - embeddings, OpenAI's
/// Responses API, a provider's own extras - with the key, the headers, the
/// error handling and `failure` of every other call. `path` is appended to
/// the provider's base URL; a whole URL is used as it is. `body` is JSON.
pub fn call(c: *Client, method: std.http.Method, path: []const u8, body: ?[]const u8) !Response {
    c.failure.clear();
    if (body != null and !method.requestHasBody())
        return c.fail(0, error.BadRequest, "a {t} request has no body", .{method});
    var arena: std.heap.ArenaAllocator = .init(c.gpa);
    defer arena.deinit();
    var ex: transport.Exchange = undefined;
    try ex.open(c, .{
        .method = method,
        .url = try c.endpoint(arena.allocator(), path),
        .payload = if (body) |b| .{ .json = b } else .none,
    });
    defer ex.deinit();
    return .{ .gpa = c.gpa, .status = ex.status, .body = try ex.readAll(c.gpa) };
}

// ---------------------------------------------------------------------------
// For the API adapters
// ---------------------------------------------------------------------------

/// `path` on the provider's base URL.
pub fn endpoint(c: *const Client, a: Allocator, path: []const u8) Allocator.Error![]const u8 {
    return transport.join(a, c.provider.base_url, path);
}

/// Record a failure in this library's own words, and return `err`.
pub fn fail(c: *Client, status: u16, err: transport.Error, comptime fmt: []const u8, args: anytype) transport.Error {
    c.failure.print(status, fmt, args);
    return err;
}

/// Record what `err` - a network error, most likely - interrupted, and
/// return it as it is.
pub fn failError(c: *Client, err: anytype, comptime fmt: []const u8, args: anytype) @TypeOf(err) {
    var buffer: [512]u8 = undefined;
    const what = std.fmt.bufPrint(&buffer, fmt, args) catch fmt;
    c.failure.print(0, "{s}: {t}", .{ what, err });
    return err;
}

/// An answer that did not have what the API promises. The provider's own
/// message wins, where there is one in the body.
pub fn invalid(c: *Client, err: error{ InvalidResponse, OutOfMemory }, root: Value, what: []const u8) error{ InvalidResponse, OutOfMemory } {
    if (err == error.OutOfMemory) return error.OutOfMemory;
    c.failure.print(0, "{s}", .{json.errorMessage(root) orelse what});
    return error.InvalidResponse;
}

/// The members of a request's `extra`, or a failure that says it is not a
/// JSON object.
pub fn extraMembers(c: *Client, a: Allocator, extra: ?[]const u8) !?*fluxion_json.Object {
    return json.parseExtra(a, extra) catch |err| switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        error.InvalidExtra => c.fail(0, error.InvalidExtra, "`extra` is not a JSON object: {s}", .{extra.?}),
    };
}

/// A JSON request body: `write(&body, args...)` between the braces, with
/// `extra` merged in.
pub fn jsonBody(c: *Client, a: Allocator, extra: ?[]const u8, comptime write: anytype, args: anytype) ![]const u8 {
    const members = try c.extraMembers(a, extra);
    return json.bodyText(a, c.gpa, members, write, args) catch error.OutOfMemory;
}

pub const JsonAnswer = struct {
    root: Value,
    /// The body as it came.
    raw: []const u8,
};

/// One exchange whose answer is JSON, read and parsed into `a`.
pub fn exchangeJson(c: *Client, a: Allocator, options: transport.Options) !JsonAnswer {
    var ex: transport.Exchange = undefined;
    try ex.open(c, options);
    defer ex.deinit();
    const raw = try ex.readAll(a);
    const root = json.parse(a, raw) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        error.InvalidResponse => return c.fail(@intFromEnum(ex.status), error.InvalidResponse, "an answer that is not JSON: {s}", .{raw[0..@min(raw.len, 300)]}),
    };
    return .{ .root = root, .raw = raw };
}
