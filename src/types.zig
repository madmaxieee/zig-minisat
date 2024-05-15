const std = @import("std");

pub const Variable = i32;

pub const Literal = struct {
    x: i32,

    pub fn init(v: Variable, b: bool) Literal {
        return .{ .x = v + v + @intFromBool(b) };
    }
    pub inline fn eql(self: Literal, p: Literal) bool {
        return std.meta.eql(self, p);
    }
    pub inline fn neq(self: Literal, p: Literal) bool {
        return !self.eql(p);
    }
    pub inline fn lt(self: Literal, p: Literal) bool {
        return self.x < p.x;
    }
    pub inline fn neg(self: Literal) Literal {
        return .{ .x = self.x ^ 1 };
    }
    pub inline fn flip(self: Literal, b: bool) Literal {
        return .{ .x = self.x ^ @intFromBool(b) };
    }
    pub inline fn sign(self: Literal) bool {
        return (self.x & 1) != 0;
    }
    pub inline fn variable(self: Literal) Variable {
        return self.x >> 1;
    }
    pub inline fn toInt(self: Literal) i32 {
        return self.x;
    }
};

pub fn literalLessThan(context: void, lhs: Literal, rhs: Literal) bool {
    _ = context;
    return lhs.lt(rhs);
}

pub const LiteralHashContext = struct {
    pub fn hash(self: LiteralHashContext, lit: Literal) u32 {
        _ = self;
        return @intCast(lit.x);
    }
    pub fn eql(self: LiteralHashContext, a: Literal, b: Literal, index: usize) bool {
        _ = self;
        _ = index;
        return a.eql(b);
    }
};

pub const lit_Undef = Literal{ .x = -2 };
pub const lit_Error = Literal{ .x = -1 };

test "Literal" {
    const testing = std.testing;
    const var_: Variable = 3;
    const lit_1: Literal = Literal.init(var_, true);
    const lit_2: Literal = lit_1.neg();
    try testing.expect(lit_1.variable() == var_);
    try testing.expect(lit_1.variable() == lit_2.variable());
    try testing.expect(lit_1.eql(lit_2.neg()));
}

pub const LiftedBool = struct {
    value: u8,

    pub inline fn fromBool(v: bool) LiftedBool {
        return LiftedBool{ .value = @intFromBool(v) };
    }
    pub inline fn eql(self: LiftedBool, b: LiftedBool) bool {
        return ((b.value & 2) & (self.value & 2)) != 0 or ((b.value & 2 == 0) and (self.value == b.value));
    }
    pub inline fn neq(self: LiftedBool, b: LiftedBool) bool {
        return !self.eql(b);
    }
    pub inline fn xor(self: LiftedBool, b: bool) LiftedBool {
        return LiftedBool{ .value = self.value ^ @intFromBool(b) };
    }
    pub inline fn @"and"(self: LiftedBool, b: LiftedBool) bool {
        const sel: u8 = (self.value << 1) | (b.value << 3);
        const v: u8 = (0xF7F755F4 >> sel) & 3;
        return LiftedBool{ .value = v };
    }
    pub inline fn @"or"(self: LiftedBool, b: LiftedBool) bool {
        const sel: u8 = (self.value << 1) | (b.value << 3);
        const v: u8 = (0xFCFCF400 >> sel) & 3;
        return LiftedBool{ .value = v };
    }
};

pub const l_True = LiftedBool{ .value = 0 };
pub const l_False = LiftedBool{ .value = 1 };
pub const l_Undef = LiftedBool{ .value = 2 };

const ClauseHeader = struct {
    mark: u32,
    learnt: bool,
    has_extra: bool,
    size: usize,
};

const ClauseData = union {
    lit: Literal,
    act: f32,
    abs: u32,
};

const ClauseError = error{
    NoExtraData,
    LiteralNotFound,
};

pub const Clause = struct {
    header: ClauseHeader,
    data: []ClauseData,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, literals: []const Literal, use_extra: bool, is_learnt: bool) !Clause {
        var clause = Clause{
            .header = .{
                .mark = 0,
                .learnt = is_learnt,
                .has_extra = use_extra,
                .size = literals.len,
            },
            .data = try allocator.alloc(ClauseData, if (use_extra) literals.len + 1 else literals.len),
            .allocator = allocator,
        };
        for (0..literals.len) |i| {
            clause.data[i] = .{ .lit = literals[i] };
        }
        if (use_extra) {
            if (is_learnt) {
                clause.data[clause.header.size] = .{ .act = 0 };
            } else {
                try clause.calcAbstraction();
            }
        }
        return clause;
    }

    pub fn deinit(self: Clause) void {
        self.allocator.free(self.data);
    }

    pub fn calcAbstraction(self: *Clause) ClauseError!void {
        if (!self.header.has_extra) {
            return ClauseError.NoExtraData;
        }
        var abs: u32 = 0;
        for (0..self.header.size) |i| {
            const bits: u32 = @intCast(self.data[i].lit.variable() & 31);
            abs |= @as(u32, 1) << @truncate(bits);
        }
        self.data[self.header.size] = .{ .abs = abs };
    }

    pub inline fn shrink(self: *Clause, i: usize) void {
        if (self.header.has_extra) {
            self.data[self.header.size - i] = self.data[self.header.size];
        }
        self.header.size -= i;
    }

    pub inline fn pop(self: *Clause) void {
        self.shrink(1);
    }

    pub inline fn mark(self: *Clause, m: u32) void {
        self.header.mark = m;
    }

    pub inline fn last(self: *Clause) Literal {
        return self.data[self.header.size - 1].lit;
    }

    pub inline fn get(self: *Clause, i: usize) Literal {
        return self.data[i].lit;
    }

    pub inline fn put(self: *Clause, i: usize, p: Literal) void {
        self.data[i] = .{ .lit = p };
    }

    pub inline fn activity(self: *Clause) ClauseError!f32 {
        if (!self.header.has_extra) {
            return ClauseError.NoExtraData;
        }
        return self.data[self.header.size].act;
    }

    pub inline fn activityPtr(self: *Clause) ClauseError!*f32 {
        if (!self.header.has_extra) {
            return ClauseError.NoExtraData;
        }
        return &self.data[self.header.size].act;
    }

    pub inline fn abstraction(self: *Clause, i: usize) ClauseError!u32 {
        if (!self.header.has_extra) {
            return ClauseError.NoExtraData;
        }
        return self.data[i].abs;
    }

    /// Checks if clause subsumes 'other', and at the same time, if it can be used to simplify 'other' by subsumption resolution.
    pub fn subsumes(self: Clause, other: Clause) union { none: void, one: Literal, all: void } {
        if (self.header.learnt or other.header.learnt) {
            return .none;
        }
        if (!self.header.has_extra or !other.header.has_extra) {
            return .none;
        }
        if (self.header.size > other.header.size or (self.abstraction().? & other.abstraction().?) != 0) {
            return .none;
        }

        var to_remove: ?Literal = null;
        const c: *Literal = self.data;
        const d: *Literal = other.data;
        outer: for (0..self.header.size) |i| {
            for (0..other.header.size) |j| {
                if (c[i].lit.eql(d[j].lit)) {
                    continue :outer;
                } else if (to_remove == null and c[i].lit.eql(d[j].lit.neg())) {
                    to_remove = c[i].lit;
                    continue :outer;
                }
            }
            // not found
            return .none;
        }

        if (to_remove == null) {
            return .all;
        } else {
            return .{ .one = to_remove.? };
        }
    }

    pub fn strengthen(self: *Clause, to_remove: Literal) ClauseError!void {
        var i: usize = 0;
        while (i < self.header.size and !to_remove.eql(self.data[i].lit)) : (i += 1) {}
        if (i >= self.header.size) {
            return ClauseError.LiteralNotFound;
        }
        while (i < self.header.size - 1) : (i += 1) {
            self.data[i] = self.data[i + 1];
        }
        self.pop();
        try self.calcAbstraction();
    }

    pub fn is_deleted(self: *Clause) bool {
        return self.header.mark == 1;
    }

    pub fn activityLessThan(context: void, a: *Clause, b: *Clause) bool {
        _ = context;
        const activity_a: f32 = a.activity() catch @panic("Clause has no activity");
        const activity_b: f32 = b.activity() catch @panic("Clause has no activity");
        return a.header.size > 2 and (b.header.size == 2 or activity_a < activity_b);
    }
};

test "Clause" {
    const testing = std.testing;
    const test_allocator = testing.allocator;
    const clause: Clause = try Clause.init(
        test_allocator,
        &[_]Literal{ Literal.init(1, true), Literal.init(2, false) },
        false,
        false,
    );
    defer clause.deinit();
    try testing.expect(clause.header.size == 2);
    try testing.expect(clause.header.learnt == false);
    try testing.expect(clause.header.has_extra == false);
}

pub fn OccList(comptime K: type, comptime V: type, comptime KHashContext: ?type) type {
    return struct {
        const Self = @This();
        const OccrMap = if (KHashContext == null)
            std.AutoArrayHashMap(K, V)
        else
            std.ArrayHashMap(K, V, KHashContext.?, false);
        const DirtyMap = if (KHashContext == null)
            std.AutoArrayHashMap(K, bool)
        else
            std.ArrayHashMap(K, bool, KHashContext.?, false);

        allocator: std.mem.Allocator,
        occs: OccrMap,
        dirty: DirtyMap,
        dirties: std.ArrayList(K),

        pub fn init(allocator: std.mem.Allocator) Self {
            return .{
                .allocator = allocator,
                .occs = OccrMap.init(allocator),
                .dirty = DirtyMap.init(allocator),
                .dirties = std.ArrayList(K).init(allocator),
            };
        }

        pub fn deinit(self: *Self) void {
            self.occs.deinit();
            self.dirty.deinit();
            self.dirties.deinit();
        }

        pub fn getPtr(self: Self, key: K) ?*V {
            return self.occs.getPtr(key);
        }

        pub fn lookup(self: *Self, key: K) !?*V {
            if (self.dirty.get(key) == false) {
                try self.dirty.put(key, true);
                try self.dirties.append(key);
            }
            return self.occs.getPtr(key);
        }

        pub fn initKey(self: *Self, key: K) void {
            const vec: ?*V = self.occs.getPtr(key);
            if (vec) |v| {
                v.* = V.init(self.allocator);
            }
        }

        pub fn clean(self: *Self, key: K) void {
            const value: *V = self.occs.getPtr(key).?;
            var j: usize = 0;
            for (0..value.items.len) |i| {
                if (!value[i].is_deleted()) {
                    self.dirties[j] = self.dirties[i];
                    j += 1;
                }
            }
            value.shrinkRetainingCapacity(value.items.len - j);
            self.dirty.put(key, false);
        }

        pub fn cleanAll(self: *Self) void {
            for (0..self.dirties.len) |i| {
                if (self.dirty.get(&self.dirties[i]) != 0) {
                    self.clean(&self.dirties[i]);
                }
            }
            self.dirties.clearRetainingCapacity();
        }

        pub fn smudge(self: *Self, key: K) void {
            if (self.dirty.get(key) == false) {
                self.dirty.put(key, true);
                self.dirties.push(key);
            }
        }

        pub fn clear(self: Self) void {
            self.occs.clearAndFree();
            self.dirty.clearAndFree();
            self.dirties.clearAndFree();
        }
    };
}
