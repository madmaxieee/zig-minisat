const std = @import("std");
const clap = @import("clap");

const io = std.io;
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

    const stderr = io.getStdErr();

    var diag = clap.Diagnostic{};
    var claps_result = clap.parse(clap.Help, &params, parsers, .{
        .diagnostic = &diag,
        .allocator = gpa,
    }) catch |err| {
        diag.report(stderr.writer(), err) catch {};
        return;
    };
    defer claps_result.deinit();

    if (claps_result.args.help != 0) {
        return clap.help(stderr.writer(), clap.Help, &params, .{});
    }

    if (claps_result.positionals.len > 1) {
        try stderr.writer().print("error: too many arguments, expected 1, got {d}\n", .{claps_result.positionals.len});
        return;
    }

    var reader: fs.File.Reader = undefined;
    // read from stdin if no file is provided
    if (claps_result.positionals.len == 0) {
        reader = io.getStdIn().reader();
    } else if (claps_result.positionals.len == 1) {
        const file_name = claps_result.positionals[0];
        const file = fs.cwd().openFile(file_name, .{}) catch |err| {
            try stderr.writer().print("error: failed to open file '{s}': {any}\n", .{ file_name, err });
            return;
        };
        reader = file.reader();
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
    const stdout = io.getStdOut();
    switch (result) {
        .sat => {
            try stdout.writer().writeAll("SATISFIABLE\n");
        },
        .unsat => {
            try stdout.writer().writeAll("UNSATISFIABLE\n");
        },
        .unknown => {
            try stdout.writer().writeAll("INDETERMINATE\n");
        },
    }
}

test {
    std.testing.refAllDeclsRecursive(@This());
}
