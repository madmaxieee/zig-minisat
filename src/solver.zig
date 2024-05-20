const std = @import("std");

const util = @import("util.zig");
const types = @import("types.zig");
const Var = types.Variable;
const Clause = types.Clause;
const Lit = types.Literal;
const Lbool = types.LiftedBool;
const OccList = types.OccList;

pub const SolverResult = enum {
    sat,
    unsat,
    unknown,
};

pub const Solver = struct {
    ptr: *anyopaque,
    vtable: VTable,

    const VTable = struct {
        deinitFn: *const fn (pointer: *anyopaque) void,
        newVarFn: *const fn (pointer: *anyopaque) anyerror!Var,
        nVarsFn: *const fn (pointer: *anyopaque) usize,
        addClauseFn: *const fn (pointer: *anyopaque, ps: []const Lit) anyerror!bool,
        solveFn: *const fn (pointer: *anyopaque) anyerror!SolverResult,
    };

    fn init(
        ptr: anytype,
        comptime deinitFn: *const fn (pointer: @TypeOf(ptr)) void,
        comptime newVarFn: *const fn (pointer: @TypeOf(ptr)) anyerror!Var,
        comptime nVarsFn: *const fn (pointer: @TypeOf(ptr)) usize,
        comptime addClauseFn: *const fn (pointer: @TypeOf(ptr), ps: []const Lit) anyerror!bool,
        comptime solveFn: *const fn (pointer: @TypeOf(ptr)) anyerror!SolverResult,
    ) Solver {
        const T = @TypeOf(ptr);
        const vtable = struct {
            pub fn deinit(pointer: *anyopaque) void {
                const self: T = @ptrCast(@alignCast(pointer));
                return deinitFn(self);
            }
            pub fn newVar(pointer: *anyopaque) anyerror!Var {
                const self: T = @ptrCast(@alignCast(pointer));
                return newVarFn(self);
            }
            pub fn nVars(pointer: *anyopaque) usize {
                const self: T = @ptrCast(@alignCast(pointer));
                return nVarsFn(self);
            }
            pub fn addClause(pointer: *anyopaque, ps: []const Lit) anyerror!bool {
                const self: T = @ptrCast(@alignCast(pointer));
                return addClauseFn(self, ps);
            }
            pub fn solve(pointer: *anyopaque) anyerror!SolverResult {
                const self: T = @ptrCast(@alignCast(pointer));
                return solveFn(self);
            }
        };
        return .{
            .ptr = ptr,
            .vtable = .{
                .deinitFn = vtable.deinit,
                .newVarFn = vtable.newVar,
                .nVarsFn = vtable.nVars,
                .addClauseFn = vtable.addClause,
                .solveFn = vtable.solve,
            },
        };
    }

    pub fn deinit(self: Solver) void {
        return self.vtable.deinitFn(self.ptr);
    }

    pub fn newVar(self: Solver) anyerror!Var {
        return self.vtable.newVarFn(self.ptr);
    }

    pub fn nVars(self: Solver) usize {
        return self.vtable.nVarsFn(self.ptr);
    }

    pub fn addClause(self: Solver, ps: []const Lit) anyerror!bool {
        return self.vtable.addClauseFn(self.ptr, ps);
    }

    pub fn solve(self: Solver) anyerror!SolverResult {
        return self.vtable.solveFn(self.ptr);
    }
};

fn VariableMap(T: type) type {
    return std.AutoArrayHashMap(Var, T);
}

const VarOrderHeapContext = struct {
    activity: *VariableMap(f64),
};

fn compareVariables(context: VarOrderHeapContext, a: Var, b: Var) std.math.Order {
    return if (context.activity.get(a).? > context.activity.get(b).?) {
        return std.math.Order.gt;
    } else if (context.activity.get(a).? < context.activity.get(b).?) {
        return std.math.Order.lt;
    } else {
        return std.math.Order.eq;
    };
}

const VarOrderHeap = std.PriorityQueue(
    Var,
    VarOrderHeapContext,
    compareVariables,
);

pub const MiniSAT = struct {
    allocator: std.mem.Allocator,
    clauseAllocator: std.heap.ArenaAllocator,

    model: std.ArrayList(Lbool),
    conflict: LiteralSet,

    verbose: bool,
    var_decay: f64,
    clause_decay: f64,
    random_var_freq: f64,
    random_seed: u64,
    luby_restart: bool,
    ccmin_mode: enum { none, basic, deep },
    phase_saving: enum { none, limited, full },
    rnd_pol: bool,
    rnd_init_act: bool,
    min_learnts_lim: u32,

    restart_first: i32,
    restart_inc: f64,
    learntsize_factor: f64,
    learntsize_inc: f64,

    learntsize_adjust_start_confl: f64,
    learntsize_adjust_inc: f64,

    solves: u64,
    starts: u64,
    decisions: u64,
    rnd_decisions: u64,
    propagations: u64,
    conflicts: u64,

    dec_vars: u64,
    num_clauses: u64,
    num_learnts: u64,
    clauses_literals: u64,
    learnt_literals: u64,
    max_literals: u64,
    tot_literals: u64,

    clauses: std.ArrayList(*Clause),
    learnts: std.ArrayList(*Clause),
    trail: std.ArrayList(Lit),
    trail_lim: std.ArrayList(usize),
    assumptions: std.ArrayList(Lit),

    activity: VariableMap(f64),
    assigns: VariableMap(Lbool),
    polarity: VariableMap(bool),
    user_pol: VariableMap(Lbool),
    decision: VariableMap(bool),
    vardata: VariableMap(VarData),
    watches: OccList(Lit, std.ArrayList(Watcher), types.LiteralHashContext),
    order_heap: VarOrderHeap,

    ok: bool,
    cla_inc: f64,
    var_inc: f64,
    qhead: usize,
    simpDB_assigns: usize,
    simpDB_props: i64,
    progress_estimate: f64,
    remove_satisfied: bool,

    next_var: Var,

    released_vars: std.ArrayList(Var),
    free_vars: std.ArrayList(Var),

    // temps
    seen: VariableMap(Seen),
    analyze_stack: std.ArrayList(AnalyzeStackElement),
    analyze_toclear: std.ArrayList(Lit),
    add_tmp: std.ArrayList(Lit),

    max_learnts: f64,
    learntsize_adjust_confl: f64,
    learntsize_adjust_cnt: i32,

    // resource constraints
    conflict_budget: ?u64,
    propagation_budget: ?u64,
    asynch_interrupt: bool,

    rand: std.Random,

    const LiteralSet = std.ArrayHashMap(Lit, void, types.LiteralHashContext, false);
    const VarData = struct {
        reason: ?*Clause,
        level: usize,
    };
    const Watcher = struct {
        clause: *Clause,
        blocker: Lit,
        pub inline fn eql(self: Watcher, other: Watcher) bool {
            return self.clause == other.clause and self.blocker == other.blocker;
        }
        pub inline fn is_deleted(self: Watcher) bool {
            return self.clause.*.header.mark == 1;
        }
    };
    const WatcherHashContext = struct {
        pub inline fn hash(self: WatcherHashContext, watcher: Watcher) u64 {
            _ = self;
            _ = watcher;
            return 0;
        }
        pub inline fn eql(self: WatcherHashContext, a: Watcher, b: Watcher) bool {
            _ = self;
            return a.eql(b);
        }
    };
    const AnalyzeStackElement = struct {
        i: usize,
        lit: Lit,
    };
    const Seen = enum { undef, source, removable, failed };

    pub fn create(allocator: std.mem.Allocator) !*MiniSAT {
        var self = try allocator.create(MiniSAT);

        var prng = std.rand.DefaultPrng.init(blk: {
            var seed: u64 = undefined;
            std.posix.getrandom(std.mem.asBytes(&seed)) catch unreachable;
            break :blk seed;
        });

        self.* = MiniSAT{
            .allocator = allocator,
            .clauseAllocator = std.heap.ArenaAllocator.init(std.heap.page_allocator),

            .model = std.ArrayList(Lbool).init(allocator),
            .conflict = LiteralSet.init(allocator),

            .verbose = true,
            .var_decay = 0.95,
            .clause_decay = 0.999,
            .random_var_freq = 0,
            .random_seed = 42,
            .luby_restart = true,
            .ccmin_mode = .deep,
            .phase_saving = .full,
            .rnd_pol = false,
            .rnd_init_act = false,
            .min_learnts_lim = 0,

            .restart_first = 100,
            .restart_inc = 2,
            .learntsize_factor = 1.0 / 3.0,
            .learntsize_inc = 1.1,

            .learntsize_adjust_start_confl = 100,
            .learntsize_adjust_inc = 1.5,

            .solves = 0,
            .starts = 0,
            .decisions = 0,
            .rnd_decisions = 0,
            .propagations = 0,
            .conflicts = 0,

            .dec_vars = 0,
            .num_clauses = 0,
            .num_learnts = 0,
            .clauses_literals = 0,
            .learnt_literals = 0,
            .max_literals = 0,
            .tot_literals = 0,

            .clauses = std.ArrayList(*Clause).init(allocator),
            .learnts = std.ArrayList(*Clause).init(allocator),
            .trail = std.ArrayList(Lit).init(allocator),
            .trail_lim = std.ArrayList(usize).init(allocator),
            .assumptions = std.ArrayList(Lit).init(allocator),

            .activity = VariableMap(f64).init(allocator),
            .assigns = VariableMap(Lbool).init(allocator),
            .polarity = VariableMap(bool).init(allocator),
            .user_pol = VariableMap(Lbool).init(allocator),
            .decision = VariableMap(bool).init(allocator),
            .vardata = VariableMap(VarData).init(allocator),
            .watches = OccList(Lit, std.ArrayList(Watcher), types.LiteralHashContext).init(allocator),
            .order_heap = VarOrderHeap.init(allocator, VarOrderHeapContext{ .activity = &self.activity }),

            .ok = true,
            .cla_inc = 1,
            .var_inc = 1,
            .qhead = 0,
            .simpDB_assigns = 0,
            .simpDB_props = 0,
            .progress_estimate = 0,
            .remove_satisfied = true,

            .next_var = 0,

            .released_vars = std.ArrayList(Var).init(allocator),
            .free_vars = std.ArrayList(Var).init(allocator),

            .seen = VariableMap(Seen).init(allocator),
            .analyze_stack = std.ArrayList(AnalyzeStackElement).init(allocator),
            .analyze_toclear = std.ArrayList(Lit).init(allocator),
            .add_tmp = std.ArrayList(Lit).init(allocator),

            .max_learnts = 0,
            .learntsize_adjust_confl = 0,
            .learntsize_adjust_cnt = 0,

            .conflict_budget = null,
            .propagation_budget = null,
            .asynch_interrupt = false,

            .rand = prng.random(),
        };

        return self;
    }

    pub fn solver(self: *MiniSAT) Solver {
        return Solver.init(
            self,
            &deinit,
            &newVar,
            &nVars,
            &addClause,
            &solve,
        );
    }

    pub fn deinit(self: *MiniSAT) void {
        self.clauseAllocator.deinit();

        self.model.deinit();
        self.conflict.deinit();

        self.clauses.deinit();
        self.learnts.deinit();
        self.trail.deinit();
        self.trail_lim.deinit();
        self.assumptions.deinit();

        self.activity.deinit();
        self.assigns.deinit();
        self.polarity.deinit();
        self.user_pol.deinit();
        self.decision.deinit();
        self.vardata.deinit();
        self.watches.deinit();
        self.order_heap.deinit();

        self.released_vars.deinit();
        self.free_vars.deinit();
        self.seen.deinit();
        self.analyze_stack.deinit();
        self.analyze_toclear.deinit();
        self.add_tmp.deinit();
    }

    fn newVar(self: *MiniSAT) !Var {
        return try self._newVar(types.l_Undef, true);
    }

    fn _newVar(self: *MiniSAT, upol: Lbool, dvar: bool) !Var {
        var v: Var = undefined;
        if (self.free_vars.items.len > 0) {
            v = self.free_vars.pop();
        } else {
            v = self.next_var;
            self.next_var += 1;
        }

        try self.watches.initKey(Lit.init(v, false));
        try self.watches.initKey(Lit.init(v, true));
        self.assigns.put(v, types.l_Undef) catch unreachable;
        self.vardata.put(v, VarData{ .reason = null, .level = 0 }) catch unreachable;
        self.activity.put(v, if (self.rnd_init_act) self.rand.float(f64) else 0) catch unreachable;
        self.seen.put(v, .undef) catch unreachable;
        self.polarity.put(v, true) catch unreachable;
        self.user_pol.put(v, upol) catch unreachable;
        self.decision.ensureTotalCapacity(@intCast(v)) catch unreachable;
        self.trail.ensureTotalCapacity(@intCast(v)) catch unreachable;

        self.setDecisionVar(v, dvar);
        return v;
    }

    fn simplify(self: *MiniSAT) !bool {
        if (self.decisionLevel() != 0) {
            @panic("simplify() called when not at decision level 0");
        }

        if (!self.ok or (try self.propagate()) != null) {
            self.ok = false;
            return false;
        }
        if (self.nAssigns() == self.simpDB_assigns or self.simpDB_props > 0) {
            return true;
        }

        try self.removeSatisfied(&self.learnts);

        if (self.remove_satisfied) {
            try self.removeSatisfied(&self.clauses);

            for (self.released_vars.items) |v| {
                if (self.seen.get(v) != .undef) {
                    @panic("seen must be false");
                }
                try self.seen.put(v, .source);
            }

            var j: usize = 0;
            for (self.trail.items) |lit| {
                if (self.seen.get(lit.variable()).? != .undef) {
                    self.trail.items[j] = lit;
                    j += 1;
                }
            }
            self.trail.shrinkRetainingCapacity(self.trail.items.len - j);
            self.qhead = self.trail.items.len;

            for (self.released_vars.items) |v| {
                try self.seen.put(v, .undef);
            }

            try self.free_vars.appendSlice(self.released_vars.items);
            self.released_vars.clearRetainingCapacity();
        }

        try self.rebuildOrderHeap();

        self.simpDB_assigns = self.nAssigns();
        self.simpDB_props = @intCast(self.clauses_literals + self.learnt_literals);

        return true;
    }

    fn solve(self: *MiniSAT) !SolverResult {
        const result = try self._solve();
        return if (result.eql(types.l_True))
            SolverResult.sat
        else if (result.eql(types.l_False))
            SolverResult.unsat
        else
            SolverResult.unknown;
    }

    fn _solve(self: *MiniSAT) !Lbool {
        self.model.clearAndFree();
        self.conflict.clearAndFree();

        if (!self.ok) {
            return types.l_False;
        }

        self.solves += 1;

        self.max_learnts = @max(
            @as(f64, @floatFromInt(self.num_clauses)) * self.learntsize_factor,
            @as(f64, @floatFromInt(self.min_learnts_lim)),
        );

        self.learntsize_adjust_confl = self.learntsize_adjust_start_confl;
        self.learntsize_adjust_cnt = @intFromFloat(self.learntsize_adjust_confl);
        var status: Lbool = types.l_Undef;

        if (self.verbose) {
            const stdout_writer = std.io.getStdOut().writer();
            try stdout_writer.writeAll(
                \\============================[ Search Statistics ]==============================
                \\| Conflicts |          ORIGINAL         |          LEARNT          | Progress |
                \\|           |    Vars  Clauses Literals |    Limit  Clauses Lit/Cl |          |
                \\===============================================================================
                \\
            );
        }

        var curr_restarts: usize = 0;
        while (status.eql(types.l_Undef)) {
            const rest_base = if (self.luby_restart)
                luby(self.restart_inc, curr_restarts)
            else
                std.math.pow(f64, self.restart_inc, @floatFromInt(curr_restarts));
            status = try self.search(@intFromFloat(rest_base * @as(f64, @floatFromInt(self.restart_first))));
            if (!self.withinBudget()) {
                break;
            }
            curr_restarts += 1;
        }

        if (self.verbose) {
            const stdout_writer = std.io.getStdOut().writer();
            try stdout_writer.writeAll("===============================================================================\n");
        }

        if (status.eql(types.l_True)) {
            try self.model.resize(self._nVars());
            for (0..self._nVars()) |i| {
                self.model.items[i] = self.varValue(@intCast(i));
            }
        } else if (status.eql(types.l_False) and self.conflict.count() == 0) {
            self.ok = false;
        }

        try self.cancelUntil(0);
        return status;
    }

    fn search(self: *MiniSAT, nof_conflicts: i32) !Lbool {
        if (!self.ok) {
            @panic("search() called when not ok");
        }

        var backtrack_level: usize = undefined;
        var conflictC: u64 = 0;
        var learnt_clause = std.ArrayList(Lit).init(self.allocator);
        defer learnt_clause.deinit();
        self.starts += 1;

        while (true) {
            const confl = try self.propagate();
            if (confl) |conflict| {
                self.conflicts += 1;
                conflictC += 1;
                if (self.decisionLevel() == 0) {
                    return types.l_False;
                }

                learnt_clause.clearRetainingCapacity();
                try self.analyze(conflict, &learnt_clause, &backtrack_level);
                try self.cancelUntil(backtrack_level);

                if (learnt_clause.items.len == 1) {
                    self.uncheckedEnqueue(learnt_clause.items[0], null);
                } else {
                    const alloc = self.clauseAllocator.allocator();
                    const c = try alloc.create(Clause);
                    c.* = try Clause.init(alloc, learnt_clause.items, true, false);
                    try self.learnts.append(c);
                    try self.attachClause(c);
                    self.claBumpActivity(c);
                    self.uncheckedEnqueue(learnt_clause.items[0], c);
                }

                self.varDecayActivity();
                self.claDecayActivity();

                if (self.learntsize_adjust_cnt == 1) {
                    self.learntsize_adjust_confl *= self.learntsize_adjust_inc;
                    self.learntsize_adjust_cnt = @intFromFloat(self.learntsize_adjust_confl);
                    self.max_learnts *= self.learntsize_inc;
                    if (self.verbose) {
                        const stdout_writer = std.io.getStdOut().writer();
                        try stdout_writer.print(
                            \\| {:9} | {:7} {:8} {:8} | {:8} {:8} {:6.0} | {:6.3} % |
                            \\
                        , .{
                            self.conflicts,

                            self.dec_vars - if (self.trail_lim.items.len == 0)
                                self.trail.items.len
                            else
                                self.trail_lim.items[0],

                            self.num_clauses,
                            self.clauses_literals,
                            self.max_learnts,

                            self.num_learnts,
                            @as(f64, @floatFromInt(self.learnt_literals)) / @as(f64, @floatFromInt(self.num_learnts)),

                            self.progressEstimate() * 100,
                        });
                    }
                } else {
                    self.learntsize_adjust_cnt -= 1;
                }
            } else {
                if ((nof_conflicts >= 0 and conflictC >= nof_conflicts) or !self.withinBudget()) {
                    self.progress_estimate = self.progressEstimate();
                    try self.cancelUntil(0);
                    return types.l_Undef;
                }

                if (self.decisionLevel() == 0 and !(try self.simplify())) {
                    return types.l_False;
                }

                if (@as(f64, @floatFromInt(self.learnts.items.len)) >= self.max_learnts + @as(f64, @floatFromInt(self.nAssigns()))) {
                    try self.reduceDB();
                }

                var next: ?Lit = null;
                while (self.decisionLevel() < self.assumptions.items.len) {
                    const p = self.assumptions.items[self.decisionLevel()];
                    if (self.litValue(p).eql(types.l_True)) {
                        try self.newDecisionLevel();
                    } else if (self.litValue(p).eql(types.l_False)) {
                        try self.analyzeFinal(p.neg(), &self.conflict);
                        return types.l_False;
                    } else {
                        next = p;
                        break;
                    }
                }

                if (next == null) {
                    // New variable decision:
                    self.decisions += 1;
                    next = self.pickBranchLit();
                    if (next == null) {
                        return types.l_True;
                    }
                }

                try self.newDecisionLevel();
                self.uncheckedEnqueue(next.?, null);
            }
        }
    }

    fn progressEstimate(self: MiniSAT) f64 {
        var progress: f64 = 0;
        const F: f64 = 1.0 / @as(f64, @floatFromInt(self._nVars()));
        for (0..self.decisionLevel()) |i| {
            const beg = if (i == 0) 0 else self.trail_lim.items[i - 1];
            const end = if (i == self.decisionLevel()) self.trail.items.len else self.trail_lim.items[i];
            progress += std.math.pow(f64, F, @floatFromInt(i)) * @as(f64, @floatFromInt(end - beg));
        }
        return progress / @as(f64, @floatFromInt(self.trail.items.len));
    }

    fn releaseVar(self: *MiniSAT, l: Lit) void {
        if (self.litValue(l).eql(types.l_Undef)) {
            addClause(.{l});
            self.released_vars.append(l.variable());
        }
    }

    fn addClause(self: *MiniSAT, ps: []const Lit) !bool {
        if (self.decisionLevel() != 0) {
            @panic("addClause() called when decisionLevel() != 0");
        }

        if (!self.ok) {
            return false;
        }

        try self.add_tmp.resize(ps.len);
        std.mem.copyForwards(Lit, self.add_tmp.items, ps);
        self.add_tmp.shrinkRetainingCapacity(ps.len);
        std.mem.sort(
            Lit,
            self.add_tmp.items,
            {},
            types.literalLessThan,
        );

        const _ps = &self.add_tmp.items;
        var j: usize = 0;
        var p: ?Lit = null;
        for (_ps.*) |p_i| {
            if (self.litValue(p_i).eql(types.l_True) or (p != null and p_i.eql(p.?.neg()))) {
                return true;
            } else if (self.litValue(p_i).neq(types.l_False) and (p != null and p_i.neq(p.?))) {
                p = p_i;
                _ps.*[j] = p_i;
                j += 1;
            }
        }
        self.add_tmp.shrinkAndFree(self.add_tmp.items.len - j);

        if (_ps.*.len == 0) {
            self.ok = false;
            return false;
        } else if (_ps.*.len == 1) {
            self.uncheckedEnqueue(_ps.*[0], null);
            self.ok = try self.propagate() == null;
            return self.ok;
        } else {
            const alloc = self.clauseAllocator.allocator();
            const c = try alloc.create(Clause);
            c.* = try Clause.init(self.clauseAllocator.allocator(), _ps.*, false, false);
            try self.clauses.append(c);
            try self.attachClause(c);
        }

        return true;
    }

    fn analyze(self: *MiniSAT, _conflict: *Clause, out_learnt: *std.ArrayList(Lit), out_btlevel: *usize) !void {
        var pathC: i32 = 0;
        var p: ?Lit = null;
        var index: usize = self.trail.items.len - 1;
        var conflict: *Clause = _conflict;

        while (true) : (pathC -= 1) {
            if (conflict.header.learnt) {
                self.claBumpActivity(conflict);
            }

            for ((if (p == null) 0 else 1)..conflict.header.size) |j| {
                const q = conflict.get(j);
                if (self.seen.get(q.variable()).? == .undef and self.level(q.variable()) > 0) {
                    try self.varBumpActivity(q.variable());
                    try self.seen.put(q.variable(), .source);
                    if (self.level(q.variable()) >= self.decisionLevel()) {
                        pathC += 1;
                    } else {
                        try out_learnt.append(q);
                    }
                }
            }

            while (self.seen.get(self.trail.items[index].variable()).? == .undef) : (index -= 1) {}

            p = self.trail.items[index + 1];
            conflict = self.reason(p.?.variable()).?;
            try self.seen.put(p.?.variable(), .undef);
            pathC -= 1;

            if (pathC <= 0) {
                break;
            }
        }
        out_learnt.items[0] = p.?.neg();

        std.mem.copyBackwards(Lit, self.analyze_toclear.items, out_learnt.items);

        // Simplify conflict clause:
        var j: usize = 0;
        switch (self.ccmin_mode) {
            .deep => {
                for (out_learnt.items) |l| {
                    if (self.reason(l.variable()) == null or !(try self.litRedundant(l))) {
                        out_learnt.items[j] = l;
                        j += 1;
                    }
                }
            },
            .basic => {
                for (out_learnt.items) |l| {
                    const v_learnt = l.variable();
                    if (self.reason(v_learnt) == null) {
                        out_learnt.items[j] = l;
                        j += 1;
                    } else {
                        const c = self.reason(v_learnt).?;
                        for (1..c.header.size) |k| {
                            const v = c.get(k).variable();
                            if (self.seen.get(v).? == .undef and self.level(v) > 0) {
                                out_learnt.items[j] = l;
                                j += 1;
                                break;
                            }
                        }
                    }
                }
            },
            .none => {
                j = out_learnt.items.len;
            },
        }

        self.max_literals += out_learnt.items.len;
        out_learnt.shrinkRetainingCapacity(out_learnt.items.len - j);
        self.tot_literals += out_learnt.items.len;

        if (out_learnt.items.len == 1) {
            out_btlevel.* = 0;
        } else {
            var max_i: usize = 1;
            for (2..out_learnt.items.len) |i| {
                if (self.level(out_learnt.items[i].variable()) > self.level(out_learnt.items[max_i].variable())) {
                    max_i = i;
                }
            }
            const p2 = out_learnt.items[max_i];
            out_learnt.items[max_i] = out_learnt.items[1];
            out_learnt.items[1] = p2;
            out_btlevel.* = self.level(p2.variable());
        }

        for (self.analyze_toclear.items) |l| {
            try self.seen.put(l.variable(), .undef);
        }
    }

    fn litRedundant(self: *MiniSAT, _p: Lit) !bool {
        var p = _p;
        if (!(self.seen.get(p.variable()).? == .undef or self.seen.get(p.variable()).? == .source)) {
            @panic("seen must be undef or source");
        }
        if (self.reason(p.variable()) == null) {
            @panic("reason must not be null");
        }

        self.analyze_stack.clearAndFree();
        var c: *Clause = self.reason(p.variable()).?;

        var i: usize = 1;
        while (true) : (i += 1) {
            if (i < c.header.size) {
                const l = c.get(i);
                if (self.level(l.variable()) == 0 or self.seen.get(l.variable()).? == .source or self.seen.get(l.variable()).? == .removable) {
                    continue;
                }
                if (self.reason(l.variable()) == null or self.seen.get(l.variable()).? == .failed) {
                    try self.analyze_stack.append(AnalyzeStackElement{ .i = 0, .lit = p });
                    for (self.analyze_stack.items) |e| {
                        if (self.seen.get(e.lit.variable()).? == .undef) {
                            try self.seen.put(e.lit.variable(), .failed);
                            try self.analyze_toclear.append(e.lit);
                        }
                    }
                    return false;
                }
                try self.analyze_stack.append(AnalyzeStackElement{ .i = i, .lit = p });
                i = 0;
                p = l;
                c = self.reason(l.variable()).?;
            } else {
                if (self.seen.get(p.variable()).? == .undef) {
                    try self.seen.put(p.variable(), .removable);
                    try self.analyze_toclear.append(p);
                }
                if (self.analyze_stack.items.len == 0) {
                    break;
                }

                i = self.analyze_stack.items[self.analyze_stack.items.len - 1].i;
                p = self.analyze_stack.items[self.analyze_stack.items.len - 1].lit;
                c = self.reason(p.variable()).?;

                _ = self.analyze_stack.pop();
            }
        }

        return true;
    }

    fn analyzeFinal(self: *MiniSAT, p: Lit, out_conflict: *LiteralSet) !void {
        out_conflict.clearAndFree();
        try out_conflict.put(p, {});

        if (self.decisionLevel() == 0) {
            return;
        }

        try self.seen.put(p.variable(), .source);

        for (self.trail_lim.items[0]..self.trail.items.len) |i| {
            const i_rev = self.trail.items.len - 1 + self.trail_lim.items[0] - i;
            const x = self.trail.items[i_rev].variable();
            if (self.seen.get(x).? != .undef) {
                if (self.reason(x) != null) {
                    if (self.level(x) <= 0) {
                        @panic("level must be > 0");
                    }
                    try out_conflict.put(self.trail.items[i_rev].neg(), {});
                } else {
                    const c = self.reason(x).?;
                    for (1..c.header.size) |j| {
                        if (self.level(c.get(j).variable()) > 0) {
                            try self.seen.put(c.get(j).variable(), .source);
                        }
                    }
                }
                try self.seen.put(x, .undef);
            }
        }

        try self.seen.put(p.variable(), .undef);
    }

    fn uncheckedEnqueue(self: *MiniSAT, p: Lit, from: ?*Clause) void {
        if (self.litValue(p).neq(types.l_Undef)) {
            @panic("literal value must be undefined");
        }
        self.assigns.put(
            p.variable(),
            Lbool.fromBool(!p.sign()),
        ) catch unreachable;
        self.vardata.put(
            p.variable(),
            VarData{
                .reason = from,
                .level = self.decisionLevel(),
            },
        ) catch unreachable;
        self.trail.append(p) catch unreachable;
    }

    fn propagate(self: *MiniSAT) !?*Clause {
        var conflict: ?*Clause = null;
        var num_props: u64 = 0;

        while (self.qhead < self.trail.items.len) {
            const p = self.trail.items[self.qhead];
            self.qhead += 1;
            const ws = (try self.watches.lookup(p)).?;
            num_props += 1;

            var i: usize = 0;
            var j: usize = 0;
            next_clause: while (i < ws.items.len) {
                const blocker = ws.*.items[i].blocker;
                if (self.litValue(blocker).eql(types.l_True)) {
                    i += 1;
                    j += 1;
                    continue;
                }

                // Make sure the false literal is data[1]:
                const c = ws.*.items[i].clause;
                const false_lit = p.neg();
                if (c.get(0).eql(false_lit)) {
                    c.put(0, c.get(1));
                    c.put(1, false_lit);
                }
                if (c.get(1).neq(false_lit)) {
                    @panic("literal 1 should be false literal");
                }
                i += 1;

                // If 0th watch is true, then clause is already satisfied.
                const first = c.get(0);
                const w = Watcher{ .clause = c, .blocker = first };
                if (!first.eql(blocker) and self.litValue(first).eql(types.l_True)) {
                    ws.*.items[j] = w;
                    j += 1;
                    continue;
                }

                // Look for new watch:
                for (2..c.header.size) |k| {
                    if (self.litValue(c.get(k)).neq(types.l_False)) {
                        c.put(1, c.get(k));
                        c.put(k, false_lit);
                        self.watches.getPtr(c.get(1).neg()).?.append(w) catch unreachable;
                        continue :next_clause;
                    }
                }

                // Did not find watch -- clause is unit under assignment:
                ws.items[j] = w;
                j += 1;
                if (self.litValue(first).eql(types.l_False)) {
                    conflict = c;
                    self.qhead = self.trail.items.len;
                    // copy the remaining watches:
                    while (i < ws.items.len) {
                        ws.*.items[j] = ws.*.items[i];
                        j += 1;
                        i += 1;
                    }
                } else {
                    self.uncheckedEnqueue(first, c);
                }
            }
            ws.shrinkAndFree(ws.items.len - j);
        }
        self.propagations += num_props;
        self.simpDB_props -= @intCast(num_props);

        return conflict;
    }

    fn attachClause(self: *MiniSAT, c: *Clause) !void {
        if (c.header.size <= 1) {
            @panic("attachClause: clause size must be > 1");
        }
        try self.watches.getPtr(c.get(0).neg()).?.append(Watcher{ .clause = c, .blocker = c.get(1) });
        try self.watches.getPtr(c.get(1).neg()).?.append(Watcher{ .clause = c, .blocker = c.get(0) });
        if (c.header.learnt) {
            self.num_learnts += 1;
            self.learnt_literals += c.header.size;
        } else {
            self.num_clauses += 1;
            self.clauses_literals += c.header.size;
        }
    }

    fn detachClause(self: *MiniSAT, c: *Clause, strict: bool) !void {
        if (c.header.size <= 1) {
            @panic("detachClause: clause size must be > 1");
        }

        if (strict) {
            const watchers_0 = self.watches.getPtr(c.get(0).neg()).?;
            const to_remove_0 = Watcher{ .clause = c, .blocker = c.get(1) };
            const index_to_remove_0 = util.index_of(Watcher, watchers_0.items, to_remove_0).?;
            _ = watchers_0.orderedRemove(index_to_remove_0);
            const watchers_1 = self.watches.getPtr(c.get(1).neg()).?;
            const to_remove_1 = Watcher{ .clause = c, .blocker = c.get(0) };
            const index_to_remove_1 = util.index_of(Watcher, watchers_1.items, to_remove_1).?;
            _ = watchers_1.orderedRemove(index_to_remove_1);
        } else {
            try self.watches.smudge(c.get(0).neg());
            try self.watches.smudge(c.get(1).neg());
        }

        if (c.header.learnt) {
            self.num_learnts -= 1;
            self.learnt_literals -= c.header.size;
        } else {
            self.num_clauses -= 1;
            self.clauses_literals -= c.header.size;
        }
    }

    fn removeClause(self: *MiniSAT, c: *Clause) !void {
        try self.detachClause(c, true);
        if (self.locked(c)) {
            self.vardata.getPtr(c.get(0).variable()).?.reason = null;
        }
        c.mark(1);
        self.clauseAllocator.allocator().destroy(c);
    }

    fn satisfied(self: *MiniSAT, c: *Clause) bool {
        for (0..c.header.size) |i| {
            if (self.litValue(c.get(i)).eql(types.l_True)) {
                return true;
            }
        }
        return false;
    }

    fn cancelUntil(self: *MiniSAT, until_level: usize) !void {
        if (self.decisionLevel() > until_level) {
            var c = self.trail.items.len - 1;
            while (c >= self.trail_lim.items[until_level]) : (c -= 1) {
                const x = self.trail.items[c].variable();
                try self.assigns.put(x, types.l_Undef);
                if (self.phase_saving == .limited or (self.phase_saving == .limited and c > self.trail_lim.items[self.trail_lim.items.len - 1])) {
                    try self.polarity.put(x, self.trail.items[c].sign());
                }
                try self.insertVarOrder(x);
                if (c == self.trail_lim.items[until_level]) {
                    break;
                }
            }
        }
    }

    fn pickBranchLit(self: *MiniSAT) ?Lit {
        var next: ?Var = null;

        if (self.rand.float(f64) < self.random_var_freq and self.order_heap.items.len != 0) {
            next = self.order_heap.items[
                @intFromFloat(self.rand.float(f64) *
                    @as(f64, @floatFromInt(self._nVars())))
            ];
            if (self.varValue(next.?).eql(types.l_Undef) and self.decision.get(next.?).?) {
                self.rnd_decisions += 1;
            }
        }

        while (next == null or self.varValue(next.?).neq(types.l_Undef) or !self.decision.get(next.?).?) {
            if (self.order_heap.len == 0) {
                next = null;
                break;
            } else {
                next = self.order_heap.remove();
            }
        }

        if (next) |next_var| {
            const next_pol = self.user_pol.get(next_var).?;
            if (next_pol.neq(types.l_Undef)) {
                return Lit.init(next_var, next_pol.eql(types.l_True));
            } else if (self.rnd_pol) {
                return Lit.init(next_var, self.rand.float(f64) < 0.5);
            } else {
                return Lit.init(next_var, self.polarity.get(next_var).?);
            }
        } else {
            return null;
        }
    }

    fn reduceDB(self: *MiniSAT) !void {
        const extra_lim: f64 = self.cla_inc - @as(f64, @floatFromInt(self.learnts.items.len));
        std.mem.sort(*Clause, self.learnts.items, {}, Clause.activityLessThan);

        var j: usize = 0;
        for (self.learnts.items, 0..) |c, i| {
            if (c.header.size > 2 and !self.locked(c) and (i < self.learnts.items.len / 2 or c.activity() < extra_lim)) {
                try self.removeClause(c);
            } else {
                self.learnts.items[j] = c;
                j += 1;
            }
        }
        self.learnts.shrinkAndFree(self.learnts.items.len - j);
    }

    fn removeSatisfied(self: *MiniSAT, clauses: *std.ArrayList(*Clause)) !void {
        var j: usize = 0;
        for (clauses.items) |c| {
            if (self.satisfied(c)) {
                try self.removeClause(c);
            } else {
                if (self.litValue(c.get(0)).neq(types.l_Undef) or
                    self.litValue(c.get(1)).neq(types.l_Undef))
                {
                    @panic("removeSatisfied: literal value must be undefined");
                }
                var k: usize = 2;
                while (k < c.header.size) : (k += 1) {
                    if (self.litValue(c.get(k)).eql(types.l_False)) {
                        c.put(k, c.get(c.header.size - 1));
                        k -= 1;
                        c.pop();
                    }
                }
                clauses.items[j] = c;
                j += 1;
            }
        }
        clauses.shrinkAndFree(clauses.items.len - j);
    }

    fn rebuildOrderHeap(self: *MiniSAT) !void {
        var heap_vars = std.ArrayList(Var).init(self.allocator);
        {
            var v: Var = 0;
            while (v < self._nVars()) : (v += 1) {
                if (self.decision.get(v) != null and self.varValue(v).eql(types.l_Undef)) {
                    try heap_vars.append(v);
                }
            }
        }

        self.order_heap.deinit();
        self.order_heap = VarOrderHeap.fromOwnedSlice(
            self.allocator,
            heap_vars.items,
            VarOrderHeapContext{ .activity = &self.activity },
        );
    }

    inline fn reason(self: MiniSAT, x: Var) ?*Clause {
        return self.vardata.get(x).?.reason;
    }

    inline fn level(self: MiniSAT, x: Var) usize {
        return self.vardata.get(x).?.level;
    }

    fn insertVarOrder(self: *MiniSAT, v: Var) !void {
        if (!self.inOrderHeap(v) and self.decision.get(v).?) {
            try self.order_heap.add(v);
        }
    }

    fn inOrderHeap(self: MiniSAT, v: Var) bool {
        for (self.order_heap.items) |x| {
            if (x == v) {
                return true;
            }
        }
        return false;
    }

    inline fn varDecayActivity(self: *MiniSAT) void {
        self.var_inc *= 1 / self.var_decay;
    }

    fn varBumpActivity(self: *MiniSAT, v: Var) !void {
        const act: *f64 = self.activity.getPtr(v).?;
        act.* += self.var_inc;
        if (act.* > 1e100) {
            for (0..self._nVars()) |i| {
                try self.activity.put(@intCast(i), act.* * 1e-100);
            }
            self.var_inc *= 1e-100;
        }
        if (self.inOrderHeap(v)) {
            try self.order_heap.update(v, v);
        }
    }

    inline fn claDecayActivity(self: *MiniSAT) void {
        self.cla_inc *= 1 / self.clause_decay;
    }

    fn claBumpActivity(self: *MiniSAT, c: *Clause) void {
        const act: *f32 = c.activityPtr();
        act.* += @floatCast(self.cla_inc);
        if (c.activity() > 1e20) {
            for (0..self.learnts.items.len) |i| {
                const act_i: *f32 = self.learnts.items[i].activityPtr();
                act_i.* *= 1e-20;
            }
            self.cla_inc *= 1e-20;
        }
    }

    inline fn enqueue(self: *MiniSAT, p: Lit, from: *Clause) bool {
        const p_val = self.litValue(p);
        if (p_val != types.l_Undef) {
            return p_val != types.l_False;
        } else {
            self.uncheckedEnqueue(p, from);
            return true;
        }
    }

    inline fn locked(self: MiniSAT, c: *Clause) bool {
        const var_0 = c.get(0).variable();
        const val_0 = self.varValue(var_0);
        return val_0.eql(types.l_True) and self.reason(var_0) == c;
    }

    inline fn newDecisionLevel(self: *MiniSAT) !void {
        try self.trail_lim.append(self.trail.items.len);
    }

    inline fn decisionLevel(self: MiniSAT) usize {
        return self.trail_lim.items.len;
    }

    inline fn abstractLevel(self: MiniSAT, x: Var) u32 {
        return 1 << (self.level(x) & 31);
    }

    inline fn litValue(self: MiniSAT, l: Lit) Lbool {
        return self.assigns.get(l.variable()).?.xor(l.sign());
    }

    inline fn varValue(self: MiniSAT, x: Var) Lbool {
        return self.assigns.get(x).?;
    }

    inline fn modelLitValue(self: MiniSAT, l: Lit) Lbool {
        return self.model.items[@intCast(l.variable())].xor(l.sign());
    }

    inline fn modelVarValue(self: MiniSAT, x: Var) Lbool {
        return self.model.items[@intCast(x)];
    }

    inline fn nAssigns(self: MiniSAT) usize {
        return self.trail.items.len;
    }

    pub fn nVars(self: *MiniSAT) usize {
        return @intCast(self.next_var);
    }

    inline fn _nVars(self: MiniSAT) usize {
        return @intCast(self.next_var);
    }

    inline fn nFreeVars(self: MiniSAT) usize {
        return self.dec_vars - (if (self.trail_lim.len == 0)
            self.trail.len
        else
            self.trail_lim.items[0]);
    }

    inline fn setPolarity(self: *MiniSAT, v: Var, b: Lbool) void {
        self.user_pol.put(v, b) catch unreachable;
    }

    inline fn setDecisionVar(self: *MiniSAT, v: Var, b: bool) void {
        if (b and !(self.decision.get(v) orelse false)) {
            self.dec_vars += 1;
        } else if (!b and (self.decision.get(v) orelse false)) {
            self.dec_vars -= 1;
        }
        self.decision.put(v, b) catch unreachable;
    }

    inline fn setConfBudget(self: *MiniSAT, x: i64) void {
        self.conflict_budget = self.conflicts + x;
    }

    inline fn setPropBudget(self: *MiniSAT, x: i64) void {
        self.propagation_budget = self.propagations + x;
    }

    inline fn interrupt(self: *MiniSAT) void {
        self.asynch_interrupt = true;
    }

    inline fn clearInterrupt(self: *MiniSAT) void {
        self.asynch_interrupt = false;
    }

    inline fn budgetOff(self: *MiniSAT) void {
        self.conflict_budget = null;
        self.propagation_budget = null;
    }

    inline fn withinBudget(self: *MiniSAT) bool {
        return (self.conflict_budget == null or self.conflicts < self.conflict_budget.?) and
            (self.propagation_budget == null or self.propagations < self.propagation_budget.?) and
            !self.asynch_interrupt;
    }
};

/// Find the finite luby subsequence that contains index 'x', and the
fn luby(y: f64, _x: usize) f64 {
    var x = _x;
    var size: usize = 1;
    var seq: usize = 0;
    while (size < x + 1) {
        seq += 1;
        size = 2 * size + 1;
    }
    while (size - 1 != x) {
        size = (size - 1) >> 1;
        seq -= 1;
        x %= size;
    }
    return std.math.pow(f64, y, @floatFromInt(seq));
}

test "MiniSAT.newVar" {
    const testing = std.testing;
    const test_allocator = testing.allocator;

    var minisat = try MiniSAT.create(test_allocator);
    defer test_allocator.destroy(minisat);
    var solver: Solver = minisat.solver();
    defer solver.deinit();

    var variables = [_]i32{0} ** 5;
    for (0..5) |i| {
        variables[i] = try solver.newVar();
    }

    try testing.expect(std.mem.eql(Var, &variables, &[_]Var{ 0, 1, 2, 3, 4 }));
}

test "MiniSAT: sat-trivial-01" {
    const testing = std.testing;
    const test_allocator = testing.allocator;

    var minisat = try MiniSAT.create(test_allocator);
    defer test_allocator.destroy(minisat);
    minisat.verbose = false;
    var solver: Solver = minisat.solver();
    defer solver.deinit();
    const num_vars = 10;
    var variables = [_]i32{0} ** num_vars;

    for (0..num_vars) |i| {
        variables[i] = try solver.newVar();
    }
    for (variables) |v| {
        _ = try solver.addClause(&[_]Lit{Lit.init(v, false)});
    }
    const result = try solver.solve();

    try testing.expect(result == .sat);
}
