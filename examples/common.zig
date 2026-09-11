// SPDX-License-Identifier: BSL-1.0

//! What every example shares: the command line, the key, the model, and
//! saying what went wrong in words a person can act on.

const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;
const ai = @import("fluxion_ai");

pub const Task = enum { chat, vision, image, video };

pub const Options = struct {
    preset: ai.Provider.Preset = .openai,
    model: ?[]const u8 = null,
    base_url: ?[]const u8 = null,
    system: ?[]const u8 = null,
    size: ?[]const u8 = null,
    aspect_ratio: ?[]const u8 = null,
    seconds: ?u32 = null,
    count: u32 = 1,
    /// `--file`: pictures to send along, or to start from.
    files: []const []const u8 = &.{},
    prompt: []const u8 = "",
};

pub const common_usage =
    \\  --provider NAME   openai (default), anthropic, deepseek, gemini, xai, groq,
    \\                    mistral, openrouter, together, fireworks, perplexity,
    \\                    ollama, lm_studio
    \\  --model NAME      the model; each provider has a default for each example
    \\  --base-url URL    any other OpenAI-compatible server
    \\
    \\The key is read from the provider's usual variable: OPENAI_API_KEY,
    \\ANTHROPIC_API_KEY, DEEPSEEK_API_KEY, GEMINI_API_KEY, XAI_API_KEY, ...
    \\
;

/// The command line, or the usage and an exit when it does not parse.
pub fn parse(init: std.process.Init, usage: []const u8) !Options {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    var options: Options = .{};
    var files: std.ArrayList([]const u8) = .empty;
    var words: std.ArrayList(u8) = .empty;
    const arena = init.arena.allocator();

    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg: []const u8 = args[i];
        if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) exitWithUsage(usage, null);
        if (std.mem.startsWith(u8, arg, "--")) {
            if (i + 1 >= args.len) exitWithUsage(usage, arg);
            i += 1;
            const value: []const u8 = args[i];
            const name = arg[2..];
            if (std.mem.eql(u8, name, "provider")) {
                options.preset = std.meta.stringToEnum(ai.Provider.Preset, value) orelse exitWithUsage(usage, value);
            } else if (std.mem.eql(u8, name, "model")) {
                options.model = value;
            } else if (std.mem.eql(u8, name, "base-url")) {
                options.base_url = value;
            } else if (std.mem.eql(u8, name, "system")) {
                options.system = value;
            } else if (std.mem.eql(u8, name, "size")) {
                options.size = value;
            } else if (std.mem.eql(u8, name, "aspect")) {
                options.aspect_ratio = value;
            } else if (std.mem.eql(u8, name, "seconds")) {
                options.seconds = std.fmt.parseInt(u32, value, 10) catch exitWithUsage(usage, value);
            } else if (std.mem.eql(u8, name, "count")) {
                options.count = std.fmt.parseInt(u32, value, 10) catch exitWithUsage(usage, value);
            } else if (std.mem.eql(u8, name, "file")) {
                try files.append(arena, value);
            } else exitWithUsage(usage, arg);
            continue;
        }
        if (words.items.len > 0) try words.append(arena, ' ');
        try words.appendSlice(arena, arg);
    }
    options.files = files.items;
    options.prompt = words.items;
    return options;
}

fn exitWithUsage(usage: []const u8, wrong: ?[]const u8) noreturn {
    if (wrong) |w| std.debug.print("not understood: {s}\n\n", .{w});
    std.debug.print("{s}\n{s}", .{ usage, common_usage });
    std.process.exit(2);
}

/// The provider the options name, with its key from the environment - or
/// any OpenAI-compatible server at `--base-url`.
pub fn provider(init: std.process.Init, options: Options) ai.Provider {
    if (options.base_url) |url| {
        const key = if (options.preset.envVar()) |name| init.environ_map.get(name) else null;
        return .compatible(url, key);
    }
    return ai.Provider.fromEnvironment(options.preset, init.environ_map) catch {
        std.debug.print("no key for {t}: set {s}\n", .{ options.preset, options.preset.envVar().? });
        std.process.exit(1);
    };
}

/// What each provider would be asked for each task, when `--model` says
/// nothing. Models come and go faster than this file; `zig build models`
/// lists what a key can use today.
pub fn defaultModel(preset: ai.Provider.Preset, task: Task) ?[]const u8 {
    return switch (task) {
        .chat => switch (preset) {
            .openai => "gpt-5-mini",
            .anthropic => "claude-sonnet-5",
            .deepseek => "deepseek-flash",
            .gemini => "gemini-3.5-flash",
            .mistral => "mistral-small-latest",
            .openrouter => "openrouter/auto",
            .perplexity => "sonar",
            .groq => "llama-3.3-70b-versatile",
            .ollama => "llama3.2",
            else => null,
        },
        .vision => switch (preset) {
            .openai => "gpt-5-mini",
            .anthropic => "claude-sonnet-5",
            .gemini => "gemini-3.5-flash",
            .mistral => "mistral-small-latest",
            .openrouter => "openrouter/auto",
            .ollama => "gemma3",
            else => null,
        },
        .image => switch (preset) {
            .openai => "gpt-image-1",
            .gemini => "gemini-3.1-flash-image",
            .xai => "grok-imagine-image-2.0",
            else => null,
        },
        .video => switch (preset) {
            .openai => "sora-2",
            .gemini => "veo-3.1-fast-generate-preview",
            .xai => "grok-imagine-video-1.5",
            else => null,
        },
    };
}

/// `--model`, or the default - or, when there is neither, the models the
/// key can use, printed, and null.
pub fn model(client: *ai.Client, options: Options, task: Task) ?[]const u8 {
    if (options.model) |m| return m;
    if (defaultModel(options.preset, task)) |m| return m;
    std.debug.print("no default {t} model for {t}; pass --model. This key can use:\n", .{ task, options.preset });
    var models = client.listModels() catch |err| {
        explain(client, err);
        return null;
    };
    defer models.deinit();
    for (models.items) |m| std.debug.print("  {s}\n", .{m.id});
    return null;
}

/// The error, what the provider said about it, and what to try.
pub fn explain(client: *const ai.Client, err: anyerror) void {
    std.debug.print("error: {t}", .{err});
    if (client.failure.status != 0) std.debug.print(" (HTTP {d})", .{client.failure.status});
    std.debug.print("\n", .{});
    if (client.failure.message().len > 0) std.debug.print("  {s}\n", .{client.failure.message()});
    const hint: ?[]const u8 = switch (err) {
        error.Unauthorized => "the key was refused - check the variable it came from",
        error.NotFound => "no such model here? `zig build models -- --provider ...` lists the ones this key can use",
        error.OutOfCredit => "the account is out of credit",
        error.RateLimited => "too many requests - wait a little and try again",
        error.Unsupported => "this provider's API cannot do that - try another --provider",
        error.ConnectionRefused, error.Unexpected => "is the server running, and is the address right?",
        else => null,
    };
    if (hint) |h| std.debug.print("  ({s})\n", .{h});
}

/// The Windows console shows UTF-8 as UTF-8 only when told to, and answers
/// come in every language. Put back what was there on the way out.
pub const Console = struct {
    previous: c_uint = 0,

    const kernel32 = struct {
        extern "kernel32" fn GetConsoleOutputCP() callconv(.winapi) c_uint;
        extern "kernel32" fn SetConsoleOutputCP(code_page: c_uint) callconv(.winapi) c_int;
    };

    pub fn utf8() Console {
        if (builtin.os.tag != .windows) return .{};
        const previous = kernel32.GetConsoleOutputCP();
        _ = kernel32.SetConsoleOutputCP(65001);
        return .{ .previous = previous };
    }

    pub fn restore(console: Console) void {
        if (builtin.os.tag != .windows) return;
        if (console.previous != 0) _ = kernel32.SetConsoleOutputCP(console.previous);
    }
};

/// Where the examples put what they make.
pub fn outputDir(io: Io) ![]const u8 {
    try Io.Dir.cwd().createDirPath(io, "zig-out");
    return "zig-out";
}
