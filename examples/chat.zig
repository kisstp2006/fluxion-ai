// SPDX-License-Identifier: BSL-1.0

//! Ask once, and print the answer.
//!
//!     zig build chat -- --provider deepseek "Why is the sky blue?"

const std = @import("std");
const Io = std.Io;
const ai = @import("fluxion_ai");
const common = @import("common.zig");

const usage =
    \\usage: zig build chat -- [options] PROMPT
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

    var answer = client.chat(.{
        .model = model,
        .system = options.system,
        .messages = &.{.user(options.prompt)},
    }) catch |err| {
        common.explain(&client, err);
        std.process.exit(1);
    };
    defer answer.deinit();

    var buffer: [4096]u8 = undefined;
    var stdout: Io.File.Writer = .init(.stdout(), init.io, &buffer);
    const out = &stdout.interface;
    if (answer.reasoning.len > 0) try out.print("[thinking]\n{s}\n[/thinking]\n\n", .{answer.reasoning});
    try out.print("{s}\n", .{answer.text});
    try out.flush();

    std.debug.print("\n-- {s}, {t} ({s}), {?d} tokens in, {?d} out\n", .{
        answer.model, answer.finish, answer.finish_reason, answer.usage.input_tokens, answer.usage.output_tokens,
    });
}
