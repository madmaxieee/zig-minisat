const std = @import("std");
const fs = std.fs;

const Solver = @import("solver.zig").Solver;
const types = @import("types.zig");
const Lit = types.Literal;

const DimacsHeader = struct {
    num_variables: usize,
    num_clauses: usize,
};

pub const DimcasParser = struct {
    allocator: std.mem.Allocator,
    literals: std.ArrayList(Lit),
    solver: *Solver,

    pub fn init(allocator: std.mem.Allocator, _solver: *Solver) DimcasParser {
        return DimcasParser{
            .allocator = allocator,
            .literals = std.ArrayList(Lit).init(allocator),
            .solver = _solver,
        };
    }

    pub fn deinit(self: DimcasParser) void {
        self.literals.deinit();
    }

    pub fn parse(self: *DimcasParser, reader: std.fs.File.Reader) !void {
        var buf_reader = std.io.bufferedReader(reader);
        var in_stream = buf_reader.reader();

        var header: ?DimacsHeader = null;
        var buf: [4096]u8 = undefined;
        var count: usize = 0;
        while (try in_stream.readUntilDelimiterOrEof(&buf, '\n')) |line| {
            const trimmed_line = std.mem.trim(u8, line, "\n");
            if (std.mem.eql(u8, trimmed_line, "%")) {
                break;
            }
            if (try self.parseComment(trimmed_line)) continue;
            if (header == null) {
                header = try self.parseHeader(trimmed_line);
            } else {
                try self.parseClause(trimmed_line);
                count += 1;
            }
        }
        if (header == null) {
            @panic("no header found");
        }
        if (header.?.num_clauses != count) {
            @panic("clause count mismatch");
        }
    }

    fn parseComment(self: DimcasParser, line: []const u8) !bool {
        _ = self;
        var it = std.mem.split(u8, line, " ");
        if (it.next()) |first_tok| {
            if (std.mem.eql(u8, first_tok, "c")) {
                return true;
            }
        }
        return false;
    }

    fn parseHeader(self: DimcasParser, line: []const u8) !DimacsHeader {
        _ = self;
        var header = DimacsHeader{
            .num_variables = undefined,
            .num_clauses = undefined,
        };
        var it = std.mem.split(u8, line, " ");
        var tok: ?[]const u8 = it.next();
        while (tok != null and tok.?.len == 0) : (tok = it.next()) {}
        if (tok) |first_tok| {
            if (!std.mem.eql(u8, first_tok, "p")) {
                @panic("expected 'p' token in header");
            }
        }
        tok = it.next();
        while (tok != null and tok.?.len == 0) : (tok = it.next()) {}
        if (tok) |second_tok| {
            if (!std.mem.eql(u8, second_tok, "cnf")) {
                @panic("expected 'cnf' token in header");
            }
        }
        tok = it.next();
        while (tok != null and tok.?.len == 0) : (tok = it.next()) {}
        if (tok) |third_tok| {
            const num_variables = std.fmt.parseInt(usize, third_tok, 10) catch @panic("invalid variable count");
            header.num_variables = num_variables;
        }
        tok = it.next();
        while (tok != null and tok.?.len == 0) : (tok = it.next()) {}
        if (tok) |fourth_tok| {
            const num_clauses = std.fmt.parseInt(usize, fourth_tok, 10) catch @panic("invalid clause count");
            header.num_clauses = num_clauses;
        }
        tok = it.next();
        while (tok != null and tok.?.len == 0) : (tok = it.next()) {}
        if (tok) |_| {
            @panic("unexpected token in header");
        }
        return header;
    }

    fn parseClause(self: *DimcasParser, line: []const u8) !void {
        self.literals.clearRetainingCapacity();
        var it = std.mem.split(u8, line, " ");
        while (it.next()) |tok| {
            if (tok.len == 0) {
                continue;
            }
            const raw_var = try std.fmt.parseInt(types.Variable, tok, 10);
            if (raw_var == 0) {
                break;
            } else if (raw_var < 0) {
                const variable = -raw_var - 1;
                try self.literals.append(Lit.init(variable, true));
                while (self.solver.nVars() <= variable) {
                    _ = try self.solver.newVar();
                }
            } else {
                const variable = raw_var - 1;
                try self.literals.append(Lit.init(variable, true));
                while (self.solver.nVars() <= variable) {
                    _ = try self.solver.newVar();
                }
            }
        }
        _ = try self.solver.addClause(self.literals.items);
    }
};
