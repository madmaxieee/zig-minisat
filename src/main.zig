const std = @import("std");
const clap = @import("clap");

const fs = std.fs;

const MiniSAT = @import("solver.zig").MiniSAT;
const Solver = @import("solver.zig").Solver;
const DimacsParser = @import("dimacs.zig").DimcasParser;

pub fn main() !void {
    var gpa_impl = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa_impl.deinit();

    const gpa = gpa_impl.allocator();

    const paramsConfig =
        \\-h, --help             Display this help and exit.
        \\-q, --quiet            Suppress all output except the result.
        \\<FILE>                 a plain text DIMCAS file.
        \\
    ;

    const params = comptime clap.parseParamsComptime(paramsConfig);

    const parsers = comptime .{
        .FILE = clap.parsers.string,
    };

    const stderr = fs.File.stderr();

    var diag = clap.Diagnostic{};
    var claps_result = clap.parse(clap.Help, &params, parsers, .{
        .diagnostic = &diag,
        .allocator = gpa,
    }) catch |err| {
        diag.reportToFile(stderr, err) catch {};
        return;
    };
    defer claps_result.deinit();

    if (claps_result.args.help != 0) {
        return clap.helpToFile(stderr, clap.Help, &params, .{});
    }

    if (claps_result.positionals.len > 1) {
        var buf: [1024]u8 = undefined;
        var writer = stderr.writer(&buf);
        try writer.interface.print("error: too many arguments, expected 1, got {d}\n", .{claps_result.positionals.len});
        try writer.interface.flush();
        return;
    }

    var stdin_buf: [4096]u8 = undefined;
    var file_buf: [4096]u8 = undefined;
    var reader: fs.File.Reader = undefined;
    // read from stdin if no file is provided
    if (claps_result.positionals.len == 0) {
        reader = fs.File.stdin().reader(&stdin_buf);
    } else if (claps_result.positionals.len == 1) {
        const file_name = claps_result.positionals[0].?;
        const file = fs.cwd().openFile(file_name, .{}) catch |err| {
            var buf: [1024]u8 = undefined;
            var writer = stderr.writer(&buf);
            writer.interface.print("error: failed to open file '{s}': {any}\n", .{ file_name, err }) catch {};
            writer.interface.flush() catch {};
            return;
        };
        reader = file.reader(&file_buf);
    } else {
        unreachable;
    }

    var minisat = try MiniSAT.create(gpa);
    minisat.verbose = claps_result.args.quiet == 0;
    defer gpa.destroy(minisat);
    var solver: Solver = minisat.solver();
    defer solver.deinit();

    var parser = DimacsParser.init(gpa, &solver);
    defer parser.deinit();
    try parser.parse(reader);

    const result = try solver.solve();
    const stdout = fs.File.stdout();
    var out_buf: [1024]u8 = undefined;
    var out_writer = stdout.writer(&out_buf);
    switch (result) {
        .sat => {
            try out_writer.interface.writeAll("SATISFIABLE\n");
        },
        .unsat => {
            try out_writer.interface.writeAll("UNSATISFIABLE\n");
        },
        .unknown => {
            try out_writer.interface.writeAll("INDETERMINATE\n");
        },
    }
    try out_writer.interface.flush();
}

test {
    std.testing.refAllDeclsRecursive(@This());
}
