// SPDX-License-Identifier: BSL-1.0

const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // The importable module. Consumers do:
    //   const ai = @import("fluxion_ai");
    // Besides the standard library it needs fluxion-json, and nothing else.
    const json_dep = b.dependency("fluxion_json", .{ .target = target, .optimize = optimize });
    const mod = b.addModule("fluxion_ai", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "fluxion_json", .module = json_dep.module("fluxion_json") }},
    });

    // zig build test: the library's own tests, then the whole way down
    // against servers on this machine.
    const test_step = b.step("test", "Run the unit tests and the local-server tests");
    const unit_tests = b.addTest(.{ .name = "fluxion-ai-tests", .root_module = mod });
    test_step.dependOn(&b.addRunArtifact(unit_tests).step);
    const mock_tests = b.addTest(.{
        .name = "fluxion-ai-mock-tests",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/mock_test.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "fluxion_ai", .module = mod }},
        }),
    });
    test_step.dependOn(&b.addRunArtifact(mock_tests).step);

    // zig build docs -> zig-out/docs
    const docs_lib = b.addLibrary(.{ .name = "fluxion-ai", .root_module = mod });
    const install_docs = b.addInstallDirectory(.{
        .source_dir = docs_lib.getEmittedDocs(),
        .install_dir = .prefix,
        .install_subdir = "docs",
    });
    const docs_step = b.step("docs", "Generate API documentation into zig-out/docs");
    docs_step.dependOn(&install_docs.step);

    // The examples, only when this is the package being built: a program
    // that depends on fluxion-ai runs this function too, and must not be
    // made to fetch what only the examples use.
    if (b.pkg_hash.len != 0) return;

    const examples_step = b.step("examples", "Build every example");
    const Example = struct { name: []const u8, description: []const u8, pictures: bool };
    const examples = [_]Example{
        .{ .name = "chat", .description = "Ask once, print the answer", .pictures = false },
        .{ .name = "stream", .description = "Print an answer as it is written", .pictures = false },
        .{ .name = "models", .description = "List the models a key can use", .pictures = false },
        .{ .name = "image", .description = "Draw a picture, save it, and look inside it", .pictures = true },
        .{ .name = "vision", .description = "Draw a test card, and ask a model what it sees", .pictures = true },
        .{ .name = "video", .description = "Make a video, wait for it, save it", .pictures = true },
    };

    // fluxion-image, for the examples that look at pictures. Lazy: it is
    // fetched the first time one of them is configured, and never for the
    // library.
    const image_dep = b.lazyDependency("fluxion_image", .{ .target = target, .optimize = optimize });

    for (examples) |example| {
        if (example.pictures and image_dep == null) continue;
        const example_mod = b.createModule(.{
            .root_source_file = b.path(b.fmt("examples/{s}.zig", .{example.name})),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "fluxion_ai", .module = mod }},
        });
        if (example.pictures) example_mod.addImport("fluxion_image", image_dep.?.module("fluxion_image"));

        const exe = b.addExecutable(.{ .name = b.fmt("fluxion-ai-{s}", .{example.name}), .root_module = example_mod });
        const install = b.addInstallArtifact(exe, .{});
        examples_step.dependOn(&install.step);
        b.getInstallStep().dependOn(&install.step);

        // zig build <name> -- [arguments]
        const run = b.addRunArtifact(exe);
        run.step.dependOn(&install.step);
        if (b.args) |args| run.addArgs(args);
        b.step(example.name, example.description).dependOn(&run.step);
    }
}
