// SPDX-License-Identifier: BSL-1.0

//! Print an answer as it is written. The thinking, where a model shows it
//! (DeepSeek, Claude with thinking on, Gemini), goes to stderr, so that
//! stdout is the answer alone.
//!
//!     zig build stream -- --provider anthropic "Write a haiku about Zig."

const std = @import("std");
const Io = std.Io;
const ai = @import("fluxion_ai");
const common = @import("common.zig");

const usage =
    \\usage: zig build stream -- [options] PROMPT
    \\  --system TEXT     instructions ahead of the prompt
;

pub fn main(init: std.process.Init) !void {
    const console: common.Console = .utf8();
    defer console.restore();

    const options = try common.parse(init, usage);
    if (options.prompt.len == 0) {
        std.debug.print("{s}\n{s}", .{ usage, common.common_usage });
        std.process.exit(2);
    }

    var client: ai.Client = .init(init.gpa, init.io, common.provider(init, options));
    defer client.deinit();
    const model = common.model(&client, options, .chat) orelse std.process.exit(1);

    const stream = client.stream(.{
        .model = model,
        .system = options.system,
        .messages = &.{.user(options.prompt)},
    }) catch |err| {
        common.explain(&client, err);
        std.process.exit(1);
    };
    defer stream.deinit();

    var buffer: [256]u8 = undefined;
    var stdout: Io.File.Writer = .init(.stdout(), init.io, &buffer);
    const out = &stdout.interface;

    while (true) {
        const event = stream.next() catch |err| {
            try out.flush();
            std.debug.print("\n", .{});
            common.explain(&client, err);
            std.process.exit(1);
        } orelse break;
        switch (event) {
            .text => |words| {
                try out.writeAll(words);
                try out.flush();
            },
            .reasoning => |thought| std.debug.print("{s}", .{thought}),
            .image => |image| std.debug.print("\n[a picture: {s}, {d} bytes]\n", .{ image.mime_type, image.bytes.len }),
        }
    }
    try out.writeAll("\n");
    try out.flush();

    std.debug.print("\n-- {s}, {t}, {?d} tokens in, {?d} out\n", .{
        stream.model, stream.finish, stream.usage.input_tokens, stream.usage.output_tokens,
    });
}
