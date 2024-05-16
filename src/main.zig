const std = @import("std");
const clap = @import("clap");

const debug = std.debug;
const io = std.io;
const fs = std.fs;

const MiniSAT = @import("solver.zig").MiniSAT;
const Solver = @import("solver.zig").Solver;

const types = @import("types.zig");
const Lit = types.Literal;

pub fn main() !void {
    var gpa_impl = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa_impl.deinit();

    const gpa = gpa_impl.allocator();

    const paramsConfig =
        \\-h, --help             Display this help and exit.
        \\<FILE>                 a plain text DIMCAS file.
        \\
    ;

    const params = comptime clap.parseParamsComptime(paramsConfig);

    const parsers = comptime .{
        .FILE = clap.parsers.string,
    };

    const stderr = io.getStdErr();

    var diag = clap.Diagnostic{};
    var res = clap.parse(clap.Help, &params, parsers, .{
        .diagnostic = &diag,
        .allocator = gpa,
    }) catch |err| {
        diag.report(stderr.writer(), err) catch {};
        return;
    };
    defer res.deinit();

    if (res.args.help != 0) {
        return clap.help(stderr.writer(), clap.Help, &params, .{});
    }

    if (res.positionals.len > 1) {
        try stderr.writer().print("error: too many arguments, expected 1, got {d}\n", .{res.positionals.len});
        return;
    }

    var reader: fs.File.Reader = undefined;
    // read from stdin if no file is provided
    if (res.positionals.len == 0) {
        reader = io.getStdIn().reader();
    } else if (res.positionals.len == 1) {
        const file_name = res.positionals[0];
        const file = fs.cwd().openFile(file_name, .{}) catch |err| {
            try stderr.writer().print("error: failed to open file '{s}': {any}\n", .{ file_name, err });
            return;
        };
        reader = file.reader();
    } else {
        unreachable;
    }

    // read the file and print it to stdout
    var buf: [4096]u8 = undefined;
    while (true) {
        const read = try reader.read(buf[0..]);
        if (read == 0) {
            break;
        }
        _ = try io.getStdOut().write(buf[0..read]);
    }

    var minisat = try MiniSAT.create(gpa);
    defer gpa.destroy(minisat);
    var solver: Solver = minisat.solver();
    defer solver.deinit();

    const num_vars = 10;
    var variables = [_]i32{0} ** num_vars;
    for (0..num_vars) |i| {
        variables[i] = try solver.newVar();
    }
    for (variables) |v| {
        debug.print("{d} ", .{v});
    }
    debug.print("\n", .{});
    for (variables) |v| {
        _ = try solver.addClause(&[_]Lit{Lit.init(v, false)});
    }
    const result = try solver.solve();
    debug.print("result: {}\n", .{result});
}

test {
    std.testing.refAllDeclsRecursive(@This());
}
