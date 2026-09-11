// SPDX-License-Identifier: BSL-1.0

//! List the models a key can use.
//!
//!     zig build models -- --provider gemini

const std = @import("std");
const Io = std.Io;
const ai = @import("fluxion_ai");
const common = @import("common.zig");

const usage = "usage: zig build models -- [options]";

pub fn main(init: std.process.Init) !void {
    const console: common.Console = .utf8();
    defer console.restore();

    const options = try common.parse(init, usage);
    var client: ai.Client = .init(init.gpa, init.io, common.provider(init, options));
    defer client.deinit();

    var models = client.listModels() catch |err| {
        common.explain(&client, err);
        std.process.exit(1);
    };
    defer models.deinit();

    var buffer: [4096]u8 = undefined;
    var stdout: Io.File.Writer = .init(.stdout(), init.io, &buffer);
    const out = &stdout.interface;
    for (models.items) |model| {
        if (std.mem.eql(u8, model.id, model.name)) {
            try out.print("{s}\n", .{model.id});
        } else {
            try out.print("{s}  ({s})\n", .{ model.id, model.name });
        }
    }
    try out.flush();
    std.debug.print("-- {d} models\n", .{models.items.len});
}
