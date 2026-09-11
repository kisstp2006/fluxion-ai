// SPDX-License-Identifier: BSL-1.0

//! Server-sent events: the way every provider streams an answer.
//!
//! A stream is lines. `data:` lines pile up, a blank line hands them over as
//! one event, `event:` names it, and a line that begins with a colon is a
//! comment - a keep-alive, usually - and is dropped. That is all of the
//! format an answer needs; `id:` and `retry:` are for a browser reconnecting,
//! which a program that asked one question does not do.
//!
//! The parser takes whole lines and knows nothing of sockets, so the same
//! code reads a live stream and the strings in the tests below.

const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Event = struct {
    /// What `event:` said, or empty. Anthropic names every event; OpenAI and
    /// Gemini name none and put everything in `data`.
    name: []const u8,
    /// The event's `data:` lines, joined by newlines.
    data: []const u8,
};

pub const Parser = struct {
    name: std.ArrayList(u8) = .empty,
    data: std.ArrayList(u8) = .empty,
    has_data: bool = false,
    /// An event was handed out, and its memory is cleared on the next line.
    dispatched: bool = false,

    pub fn deinit(p: *Parser, gpa: Allocator) void {
        p.name.deinit(gpa);
        p.data.deinit(gpa);
        p.* = undefined;
    }

    /// One line, without the `\n` that ended it; a `\r` before it is dropped
    /// here. Returns the event a blank line completes. Its slices live until
    /// the next call.
    pub fn feed(p: *Parser, gpa: Allocator, raw_line: []const u8) Allocator.Error!?Event {
        if (p.dispatched) p.clear();

        const line = if (std.mem.endsWith(u8, raw_line, "\r")) raw_line[0 .. raw_line.len - 1] else raw_line;
        if (line.len == 0) {
            // A blank line with nothing gathered dispatches nothing, and
            // forgets any name that was set for it.
            if (!p.has_data) {
                p.name.clearRetainingCapacity();
                return null;
            }
            p.dispatched = true;
            return .{ .name = p.name.items, .data = p.data.items };
        }
        if (line[0] == ':') return null;

        const colon = std.mem.findScalar(u8, line, ':');
        const field = if (colon) |i| line[0..i] else line;
        var value: []const u8 = if (colon) |i| line[i + 1 ..] else "";
        if (std.mem.startsWith(u8, value, " ")) value = value[1..];

        if (std.mem.eql(u8, field, "data")) {
            if (p.has_data) try p.data.append(gpa, '\n');
            try p.data.appendSlice(gpa, value);
            p.has_data = true;
        } else if (std.mem.eql(u8, field, "event")) {
            p.name.clearRetainingCapacity();
            try p.name.appendSlice(gpa, value);
        }
        return null;
    }

    /// At the end of the stream: an event the server never closed with a
    /// blank line. The standard throws it away; a server that forgets the
    /// last blank line is common enough that this hands it over instead.
    pub fn finish(p: *Parser) ?Event {
        if (p.dispatched or !p.has_data) return null;
        p.dispatched = true;
        return .{ .name = p.name.items, .data = p.data.items };
    }

    fn clear(p: *Parser) void {
        p.name.clearRetainingCapacity();
        p.data.clearRetainingCapacity();
        p.has_data = false;
        p.dispatched = false;
    }
};

fn expectEvents(input: []const u8, expected: []const Event) !void {
    const gpa = std.testing.allocator;
    var parser: Parser = .{};
    defer parser.deinit(gpa);

    var seen: usize = 0;
    var lines = std.mem.splitScalar(u8, input, '\n');
    while (lines.next()) |line| {
        if (try parser.feed(gpa, line)) |event| {
            try std.testing.expect(seen < expected.len);
            try std.testing.expectEqualStrings(expected[seen].name, event.name);
            try std.testing.expectEqualStrings(expected[seen].data, event.data);
            seen += 1;
        }
    }
    if (parser.finish()) |event| {
        try std.testing.expect(seen < expected.len);
        try std.testing.expectEqualStrings(expected[seen].data, event.data);
        seen += 1;
    }
    try std.testing.expectEqual(expected.len, seen);
}

test "OpenAI-shaped: data only, closed by [DONE]" {
    try expectEvents(
        "data: {\"a\":1}\n\ndata: {\"a\":2}\n\ndata: [DONE]\n\n",
        &.{
            .{ .name = "", .data = "{\"a\":1}" },
            .{ .name = "", .data = "{\"a\":2}" },
            .{ .name = "", .data = "[DONE]" },
        },
    );
}

test "Anthropic-shaped: named events, CRLF, and pings" {
    try expectEvents(
        "event: message_start\r\ndata: {\"type\":\"message_start\"}\r\n\r\n" ++
            "event: ping\r\ndata: {\"type\": \"ping\"}\r\n\r\n",
        &.{
            .{ .name = "message_start", .data = "{\"type\":\"message_start\"}" },
            .{ .name = "ping", .data = "{\"type\": \"ping\"}" },
        },
    );
}

test "comments, multi-line data, no space after the colon, a missing last blank line" {
    try expectEvents(
        ": OPENROUTER PROCESSING\n\ndata:one\ndata: two\n\n\n\nevent: x\n\ndata: last",
        &.{
            .{ .name = "", .data = "one\ntwo" },
            .{ .name = "", .data = "last" },
        },
    );
}
