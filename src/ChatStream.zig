// SPDX-License-Identifier: BSL-1.0

//! An answer arriving a few words at a time.
//!
//! ```zig
//! const stream = try client.stream(.{
//!     .model = "deepseek-flash",
//!     .messages = &.{.user("Tell me about Zig.")},
//! });
//! defer stream.deinit();
//!
//! while (try stream.next()) |event| switch (event) {
//!     .text => |words| try out.writeAll(words),
//!     .reasoning, .image => {},
//! };
//! // stream.text.items is the whole answer; stream.usage what it cost.
//! ```
//!
//! What `next` hands out lives until the next call to `next`; the answer
//! gathered so far - `text`, `reasoning`, `images` - lives as long as the
//! stream. Three APIs stream three ways, and all of them come out of here as
//! the same three kinds of event.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Io = std.Io;

const Client = @import("Client.zig");
const Provider = @import("Provider.zig");
const chat_types = @import("chat.zig");
const GeneratedImage = @import("image.zig").GeneratedImage;
const sse = @import("sse.zig");
const transport = @import("transport.zig");
const openai = @import("api/openai.zig");
const anthropic = @import("api/anthropic.zig");
const gemini = @import("api/gemini.zig");

const ChatStream = @This();

pub const Event = union(enum) {
    /// More of the answer.
    text: []const u8,
    /// More of the thinking before it, where the provider shows it.
    reasoning: []const u8,
    /// A picture in the answer, whole. Lives as long as the stream.
    image: GeneratedImage,
};

client: *Client,
api: Provider.Api,
exchange: transport.Exchange,
/// Lives as long as the stream: the pictures, and the few strings kept.
arena: std.heap.ArenaAllocator,
/// Lives for one call to `next`: the event being taken apart.
scratch: std.heap.ArenaAllocator,
parser: sse.Parser = .{},
line: Io.Writer.Allocating,
queue: std.ArrayList(Event) = .empty,
queue_index: usize = 0,
/// The provider has said everything it is going to.
done: bool = false,

/// Everything handed out as `.text` so far, joined: the whole answer, once
/// `next` has returned null.
text: std.ArrayList(u8) = .empty,
/// Everything handed out as `.reasoning`, joined.
reasoning: std.ArrayList(u8) = .empty,
images: std.ArrayList(GeneratedImage) = .empty,
finish: chat_types.Finish = .unknown,
finish_reason: []const u8 = "",
/// Filled in as the provider reports it: Anthropic at both ends, Gemini
/// with every chunk, OpenAI-shaped servers at the end if at all (see
/// `Provider.stream_usage`).
usage: chat_types.Usage = .{},
model: []const u8 = "",
id: []const u8 = "",

/// Send `request` and wait for the answer to begin. Use `Client.stream`.
pub fn open(client: *Client, request: chat_types.ChatRequest) !*ChatStream {
    const gpa = client.gpa;
    const s = try gpa.create(ChatStream);
    errdefer gpa.destroy(s);
    s.* = .{
        .client = client,
        .api = client.provider.api,
        .exchange = undefined,
        .arena = .init(gpa),
        .scratch = .init(gpa),
        .line = .init(gpa),
    };
    errdefer {
        s.arena.deinit();
        s.scratch.deinit();
        s.line.deinit();
    }

    const a = s.scratch.allocator();
    const options = switch (s.api) {
        .openai => try openai.streamOptions(client, a, request),
        .anthropic => try anthropic.streamOptions(client, a, request),
        .gemini => try gemini.streamOptions(client, a, request),
    };
    try s.exchange.open(client, options);
    return s;
}

pub fn deinit(s: *ChatStream) void {
    const gpa = s.client.gpa;
    s.exchange.deinit();
    s.parser.deinit(gpa);
    s.line.deinit();
    s.text.deinit(gpa);
    s.reasoning.deinit(gpa);
    s.scratch.deinit();
    s.arena.deinit();
    gpa.destroy(s);
}

/// The next piece of the answer, or null when there is no more. A failure
/// part way - the connection dropped, the provider reported an error in the
/// stream - is an error here, with `client.failure` saying what happened.
pub fn next(s: *ChatStream) !?Event {
    while (true) {
        if (s.queue_index < s.queue.items.len) {
            defer s.queue_index += 1;
            return s.queue.items[s.queue_index];
        }
        if (s.done) return null;

        // Everything handed out before this call is spent.
        _ = s.scratch.reset(.retain_capacity);
        s.queue = .empty;
        s.queue_index = 0;

        const maybe_line = s.readLine() catch |err| {
            s.done = true;
            return err;
        };
        const line = maybe_line orelse {
            s.done = true;
            if (s.parser.finish()) |event| try s.handle(event);
            continue;
        };
        if (try s.parser.feed(s.client.gpa, line)) |event| try s.handle(event);
    }
}

fn handle(s: *ChatStream, event: sse.Event) !void {
    const a = s.scratch.allocator();
    const result = switch (s.api) {
        .openai => openai.streamEvent(s, a, event),
        .anthropic => anthropic.streamEvent(s, a, event),
        .gemini => gemini.streamEvent(s, a, event),
    };
    result catch |err| {
        s.done = true;
        return err;
    };
}

/// One line of the body, without its newline; null at the end. A line can
/// be megabytes - a picture, inline - so it is gathered rather than peeked.
fn readLine(s: *ChatStream) !?[]const u8 {
    s.line.clearRetainingCapacity();
    const r = s.exchange.reader;
    _ = r.streamDelimiterLimit(&s.line.writer, '\n', .limited(s.client.max_response_len)) catch |err| switch (err) {
        error.StreamTooLong => return s.client.fail(0, error.ResponseTooLarge, "a line of the stream is larger than max_response_len", .{}),
        error.ReadFailed => return s.exchange.readFailure("reading the stream"),
        error.WriteFailed => return error.OutOfMemory,
    };
    // Either the newline is next, or the stream has ended.
    _ = r.takeByte() catch |err| switch (err) {
        error.EndOfStream => return if (s.line.written().len == 0) null else s.line.written(),
        error.ReadFailed => return s.exchange.readFailure("reading the stream"),
    };
    return s.line.written();
}

// ---------------------------------------------------------------------------
// For the API adapters
// ---------------------------------------------------------------------------

/// Hand `event` out next, and add it to what has been gathered.
pub fn emit(s: *ChatStream, event: Event) Allocator.Error!void {
    const gpa = s.client.gpa;
    switch (event) {
        .text => |t| try s.text.appendSlice(gpa, t),
        .reasoning => |r| try s.reasoning.appendSlice(gpa, r),
        .image => {},
    }
    try s.queue.append(s.scratch.allocator(), event);
}

/// A picture, copied out of the event's memory into the stream's.
pub fn emitImage(s: *ChatStream, image: GeneratedImage) Allocator.Error!void {
    const a = s.arena.allocator();
    const kept: GeneratedImage = .{
        .bytes = try a.dupe(u8, image.bytes),
        .mime_type = try a.dupe(u8, image.mime_type),
        .url = if (image.url) |url| try a.dupe(u8, url) else null,
        .revised_prompt = if (image.revised_prompt) |p| try a.dupe(u8, p) else null,
    };
    try s.images.append(a, kept);
    try s.emit(.{ .image = kept });
}

pub fn setFinish(s: *ChatStream, reason: []const u8, finish: chat_types.Finish) Allocator.Error!void {
    s.finish_reason = try s.keep(s.finish_reason, reason);
    s.finish = finish;
}

pub fn setModel(s: *ChatStream, model: []const u8) Allocator.Error!void {
    s.model = try s.keep(s.model, model);
}

pub fn setId(s: *ChatStream, id: []const u8) Allocator.Error!void {
    s.id = try s.keep(s.id, id);
}

/// `new`, in the stream's memory - unless it is what is there already,
/// which is the usual case for a model name repeated in every chunk.
fn keep(s: *ChatStream, old: []const u8, new: []const u8) Allocator.Error![]const u8 {
    if (std.mem.eql(u8, old, new)) return old;
    return s.arena.allocator().dupe(u8, new);
}

/// End the stream with `err`, and what the provider said in
/// `client.failure`.
pub fn fail(s: *ChatStream, err: transport.Error, comptime fmt: []const u8, args: anytype) transport.Error {
    s.done = true;
    return s.client.fail(0, err, fmt, args);
}
