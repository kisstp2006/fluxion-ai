// SPDX-License-Identifier: BSL-1.0

//! Who is asked, and how they want to be asked.
//!
//! Three shapes of API cover almost everyone. OpenAI's is the one nearly
//! every other provider copies - DeepSeek, xAI, Groq, Mistral, OpenRouter,
//! Together, Fireworks, Perplexity, and the servers that run a model on this
//! machine, Ollama and LM Studio among them. Anthropic's is the second, and
//! Google's the third. A provider is one of the three, a base URL and a key;
//! the presets fill those in, and anything else that speaks one of the
//! shapes is a `Provider` literal away:
//!
//! ```zig
//! const vllm: ai.Provider = .compatible("http://gpu-box:8000/v1", null);
//! const azure: ai.Provider = .{
//!     .api = .openai,
//!     .base_url = "https://my-resource.openai.azure.com/openai/v1",
//!     .api_key = key,
//!     .auth = .{ .header = "api-key" },
//! };
//! ```
//!
//! Nothing here is copied. The strings a provider points at - the key most
//! of all - belong to the caller and must outlive the calls made with it.

const std = @import("std");

const Provider = @This();

/// Whose API this provider speaks.
api: Api,
/// Where the API lives, up to and including its version: the paths the API
/// defines are appended to it. `https://api.openai.com/v1`,
/// `https://api.anthropic.com/v1`,
/// `https://generativelanguage.googleapis.com/v1beta`.
base_url: []const u8,
/// Null for a server that asks for none.
api_key: ?[]const u8 = null,
/// How the key is presented.
auth: Auth = .bearer,
/// Sent with every request after the ones this library writes: an
/// organisation, a project, OpenRouter's `HTTP-Referer`, a beta flag.
extra_headers: []const std.http.Header = &.{},

/// OpenAI-shaped only. OpenAI's own reasoning models refuse `max_tokens`
/// and want `max_completion_tokens`; most of the servers that copy the API
/// know only the older name.
max_tokens_field: MaxTokensField = .max_tokens,
/// OpenAI-shaped only. Ask for token counts at the end of a stream
/// (`stream_options.include_usage`). Off by default, because a strict server
/// answers a field it does not know with a 400.
stream_usage: bool = false,
/// OpenAI-shaped only. How the video endpoints are laid out.
video_style: VideoStyle = .openai,
/// Anthropic-shaped only. The `anthropic-version` header.
anthropic_version: []const u8 = "2023-06-01",

pub const Api = enum {
    /// `POST /chat/completions`, `/images/generations`, `/videos`, `/models`.
    openai,
    /// `POST /messages`, `/models`.
    anthropic,
    /// `POST /models/{model}:generateContent`, `:predict`,
    /// `:predictLongRunning`, and `/models`.
    gemini,
};

pub const Auth = union(enum) {
    /// `authorization: Bearer <key>`. OpenAI and everyone who copies it,
    /// and Vertex AI given an OAuth access token as the key.
    bearer,
    /// The key alone, in a header of its own: `x-api-key` for Anthropic,
    /// `x-goog-api-key` for Gemini, `api-key` for Azure.
    header: []const u8,
    /// No key is sent.
    none,
};

pub const MaxTokensField = enum { max_tokens, max_completion_tokens };

pub const VideoStyle = enum {
    /// OpenAI's Sora: `POST /videos` as a form, polled at `/videos/{id}`,
    /// the file at `/videos/{id}/content`.
    openai,
    /// xAI's: `POST /videos/generations` as JSON, polled at `/videos/{id}`,
    /// the file at a link in the answer.
    xai,
};

/// Providers this library knows the address of.
pub const Preset = enum {
    openai,
    anthropic,
    deepseek,
    gemini,
    xai,
    groq,
    mistral,
    openrouter,
    together,
    fireworks,
    perplexity,
    /// Ollama's OpenAI-compatible endpoint on this machine.
    ollama,
    /// LM Studio's local server.
    lm_studio,

    /// The environment variable the provider's own tools read the key from,
    /// or null for a local server that needs none.
    pub fn envVar(p: Preset) ?[]const u8 {
        return switch (p) {
            .openai => "OPENAI_API_KEY",
            .anthropic => "ANTHROPIC_API_KEY",
            .deepseek => "DEEPSEEK_API_KEY",
            .gemini => "GEMINI_API_KEY",
            .xai => "XAI_API_KEY",
            .groq => "GROQ_API_KEY",
            .mistral => "MISTRAL_API_KEY",
            .openrouter => "OPENROUTER_API_KEY",
            .together => "TOGETHER_API_KEY",
            .fireworks => "FIREWORKS_API_KEY",
            .perplexity => "PERPLEXITY_API_KEY",
            .ollama, .lm_studio => null,
        };
    }
};

/// A known provider, with `api_key`.
pub fn preset(p: Preset, api_key: ?[]const u8) Provider {
    const bearer_or_none: Auth = if (api_key != null) .bearer else .none;
    return switch (p) {
        .openai => .{
            .api = .openai,
            .base_url = "https://api.openai.com/v1",
            .api_key = api_key,
            .max_tokens_field = .max_completion_tokens,
            .stream_usage = true,
        },
        .anthropic => .{
            .api = .anthropic,
            .base_url = "https://api.anthropic.com/v1",
            .api_key = api_key,
            .auth = .{ .header = "x-api-key" },
        },
        .deepseek => .{
            .api = .openai,
            .base_url = "https://api.deepseek.com",
            .api_key = api_key,
            .stream_usage = true,
        },
        .gemini => .{
            .api = .gemini,
            .base_url = "https://generativelanguage.googleapis.com/v1beta",
            .api_key = api_key,
            .auth = .{ .header = "x-goog-api-key" },
        },
        .xai => .{
            .api = .openai,
            .base_url = "https://api.x.ai/v1",
            .api_key = api_key,
            .video_style = .xai,
        },
        .groq => .{ .api = .openai, .base_url = "https://api.groq.com/openai/v1", .api_key = api_key },
        .mistral => .{ .api = .openai, .base_url = "https://api.mistral.ai/v1", .api_key = api_key },
        .openrouter => .{ .api = .openai, .base_url = "https://openrouter.ai/api/v1", .api_key = api_key },
        .together => .{ .api = .openai, .base_url = "https://api.together.xyz/v1", .api_key = api_key },
        .fireworks => .{ .api = .openai, .base_url = "https://api.fireworks.ai/inference/v1", .api_key = api_key },
        .perplexity => .{ .api = .openai, .base_url = "https://api.perplexity.ai", .api_key = api_key },
        .ollama => .{ .api = .openai, .base_url = "http://localhost:11434/v1", .api_key = api_key, .auth = bearer_or_none },
        .lm_studio => .{ .api = .openai, .base_url = "http://localhost:1234/v1", .api_key = api_key, .auth = bearer_or_none },
    };
}

pub fn openai(api_key: []const u8) Provider {
    return preset(.openai, api_key);
}

pub fn anthropic(api_key: []const u8) Provider {
    return preset(.anthropic, api_key);
}

pub fn deepseek(api_key: []const u8) Provider {
    return preset(.deepseek, api_key);
}

pub fn gemini(api_key: []const u8) Provider {
    return preset(.gemini, api_key);
}

/// Any server that speaks OpenAI's API: vLLM, llama.cpp's server, a proxy,
/// a provider without a preset. With no key, none is sent.
pub fn compatible(base_url: []const u8, api_key: ?[]const u8) Provider {
    return .{
        .api = .openai,
        .base_url = base_url,
        .api_key = api_key,
        .auth = if (api_key != null) .bearer else .none,
    };
}

/// A preset, with its key read from the variable `Preset.envVar` names
/// (`GOOGLE_API_KEY` also does for Gemini). The key points into
/// `environ_map`, which must outlive the provider.
pub fn fromEnvironment(p: Preset, environ_map: *const std.process.Environ.Map) error{MissingApiKey}!Provider {
    const name = p.envVar() orelse return preset(p, null);
    const key = nonEmpty(environ_map.get(name)) orelse
        (if (p == .gemini) nonEmpty(environ_map.get("GOOGLE_API_KEY")) else null) orelse
        return error.MissingApiKey;
    return preset(p, key);
}

fn nonEmpty(s: ?[]const u8) ?[]const u8 {
    const v = std.mem.trim(u8, s orelse return null, " \t\r\n");
    return if (v.len == 0) null else v;
}

test preset {
    const p = preset(.anthropic, "k");
    try std.testing.expectEqual(Api.anthropic, p.api);
    try std.testing.expectEqualStrings("x-api-key", p.auth.header);

    try std.testing.expectEqual(Auth.none, preset(.ollama, null).auth);
    try std.testing.expectEqual(Auth.bearer, preset(.ollama, "k").auth);
    try std.testing.expectEqual(MaxTokensField.max_completion_tokens, openai("k").max_tokens_field);
    try std.testing.expectEqual(Auth.none, compatible("http://x/v1", null).auth);
}

test fromEnvironment {
    var map: std.process.Environ.Map = .init(std.testing.allocator);
    defer map.deinit();
    try std.testing.expectError(error.MissingApiKey, fromEnvironment(.deepseek, &map));
    try map.put("GOOGLE_API_KEY", " g-key \n");
    try std.testing.expectEqualStrings("g-key", (try fromEnvironment(.gemini, &map)).api_key.?);
    try std.testing.expectEqual(null, (try fromEnvironment(.lm_studio, &map)).api_key);
}
