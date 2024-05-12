const std = @import("std");

const types = @import("types.zig");
const Var = types.Variable;
const Clause = types.Clause;
const Lit = types.Literal;
const Lbool = types.LiftedBool;
const OccList = types.OccList;

pub const Solver = struct {
    ptr: *anyopaque,
    newVarFn: *const fn (pointer: *anyopaque, upol: Lbool, dvar: bool) Var,

    fn init(ptr: anytype) Solver {
        const T = @TypeOf(ptr);
        const ptr_info = @typeInfo(T);
        const gen = struct {
            pub fn newVarFn(pointer: *anyopaque, upol: Lbool, dvar: bool) Var {
                const self: T = @ptrCast(@alignCast(pointer));
                return ptr_info.Pointer.child.newVar(self, upol, dvar);
            }
        };
        return .{
            .ptr = ptr,
            .newVarFn = gen.newVarFn,
        };
    }

    pub fn newVar(self: Solver, upol: Lbool, dvar: bool) Var {
        return self.newVarFn(self.ptr, upol, dvar);
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
    const LiteralSet = std.ArrayHashMap(Lit, void, types.LiteralHashContext, false);
    const VarData = struct {
        reason: ?*Clause,
        level: usize,
    };
    const Watcher = struct {
        clause: *Clause,
        blocker: Lit,
        inline fn eql(self: Watcher, other: Watcher) bool {
            return self.clause == other.clause and self.blocker == other.blocker;
        }
        inline fn is_deleted(self: Watcher) bool {
            return self.clause.mark == 1;
        }
    };
    const WatcherHashContext = struct {
        inline fn hash(self: WatcherHashContext, watcher: Watcher) u64 {
            _ = self;
            _ = watcher;
            return 0;
        }
        inline fn eql(self: WatcherHashContext, a: Watcher, b: Watcher) bool {
            _ = self;
            return a.eql(b);
        }
    };
    const AnalyzeStackElement = struct {
        i: i32,
        lit: Lit,
    };

    model: std.ArrayList(Lbool),
    conflict: LiteralSet,

    verbosity: std.log.Level,
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

    learntsize_adjust_start_confl: i32,
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
    max_literals: u64,
    tot_literals: u64,

    clauses: std.ArrayList(*Clause),
    learnts: std.ArrayList(*Clause),
    trail: std.ArrayList(Lit),
    trail_lim: std.ArrayList(i32),
    assumptions: std.ArrayList(Lbool),

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
    qhead: i32,
    simpDB_assigns: i32,
    simpDB_props: i64,
    progress_estimate: f64,
    remove_satisfied: bool,

    next_var: Var,
    clauseAllocator: std.heap.ArenaAllocator,

    released_vars: std.ArrayList(Var),
    free_vars: std.ArrayList(Var),

    // temps
    seen: VariableMap(u8),
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

    pub fn create(allocator: std.mem.Allocator) !*MiniSAT {
        var self = try allocator.create(MiniSAT);

        var prng = std.rand.DefaultPrng.init(blk: {
            var seed: u64 = undefined;
            std.posix.getrandom(std.mem.asBytes(&seed)) catch unreachable;
            break :blk seed;
        });

        self.* = MiniSAT{
            .model = std.ArrayList(Lbool).init(allocator),
            .conflict = LiteralSet.init(allocator),

            .verbosity = std.log.Level.info,
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
            .max_literals = 0,
            .tot_literals = 0,

            .clauses = std.ArrayList(*Clause).init(allocator),
            .learnts = std.ArrayList(*Clause).init(allocator),
            .trail = std.ArrayList(Lit).init(allocator),
            .trail_lim = std.ArrayList(i32).init(allocator),
            .assumptions = std.ArrayList(Lbool).init(allocator),

            .activity = VariableMap(f64).init(allocator),
            .assigns = VariableMap(Lbool).init(allocator),
            .polarity = VariableMap(bool).init(allocator),
            .user_pol = VariableMap(Lbool).init(allocator),
            .decision = VariableMap(bool).init(allocator),
            .vardata = VariableMap(VarData).init(allocator),
            .watches = OccList(Lit, std.ArrayList(Watcher), types.LiteralHashContext).init(allocator),
            .order_heap = VarOrderHeap.init(allocator, VarOrderHeapContext{
                .activity = &self.activity,
            }),

            .ok = true,
            .cla_inc = 1,
            .var_inc = 1,
            .qhead = 0,
            .simpDB_assigns = -1,
            .simpDB_props = 0,
            .progress_estimate = 0,
            .remove_satisfied = true,

            .next_var = 0,
            .clauseAllocator = std.heap.ArenaAllocator.init(std.heap.page_allocator),

            .released_vars = std.ArrayList(Var).init(allocator),
            .free_vars = std.ArrayList(Var).init(allocator),

            .seen = VariableMap(u8).init(allocator),
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
        return Solver.init(self);
    }

    pub fn newVar(self: *MiniSAT, upol: Lbool, dvar: bool) Var {
        var v: Var = undefined;
        if (self.free_vars.items.len > 0) {
            v = self.free_vars.pop();
        } else {
            v = self.next_var;
            self.next_var += 1;
        }

        self.watches.initKey(Lit.init(v, false));
        self.watches.initKey(Lit.init(v, true));
        self.assigns.put(v, types.l_Undef) catch unreachable;
        self.vardata.put(v, VarData{ .reason = null, .level = 0 }) catch unreachable;
        self.activity.put(v, if (self.rnd_init_act) self.rand.float(f64) else 0) catch unreachable;
        self.seen.put(v, 0) catch unreachable;
        self.polarity.put(v, true) catch unreachable;
        self.user_pol.put(v, upol) catch unreachable;
        self.decision.ensureTotalCapacity(@intCast(v)) catch unreachable;
        self.trail.ensureTotalCapacity(@intCast(v)) catch unreachable;

        self.setDecisionVar(v, dvar);
        return v;
    }

    pub fn releaseVar(self: *MiniSAT, l: Lit) void {
        if (self.litValue(l) == types.l_Undef) {
            addClause(.{l});
            self.released_vars.append(l.variable());
        }
    }

    pub fn addClause(self: *MiniSAT, ps: []Lit) bool {
        if (self.decisionLevel() != 0) {
            unreachable;
        }

        if (!self.ok) {
            return false;
        }

        self.add_tmp.ensureTotalCapacity(ps.len) catch false;
        std.mem.copyForwards(Lit, self.add_tmp.items, ps);
        self.add_tmp.shrinkRetainingCapacity(ps.len);
        std.mem.sort(
            Lit,
            self.add_tmp.items,
            {},
            types.literalLessThan,
        );

        var _ps = self.add_tmp.items;
        var count: usize = 0;
        var p = types.lit_Undef;
        for (0..self.add_tmp.len) |i| {
            if (self.litValue(_ps.items[i]) == types.l_True or _ps.items[i] == p.neg()) {
                return true;
            } else if (self.litValue(_ps.items[i]) != types.l_False and _ps.items[i] != p) {
                p = _ps.items[i];
                _ps.items[count] = _ps.items[i];
                count += 1;
            }
        }
        self.add_tmp.shrinkAndFree(self.add_tmp.len - count);

        if (self.add_tmp.items.len == 0) {
            self.ok = false;
            return false;
        } else if (self.add_tmp.items.len == 1) {
            self.uncheckedEnqueue(_ps[0], null);
            // TODO:
        } else {
            // TODO:
        }

        return true;
    }

    fn uncheckedEnqueue(self: *MiniSAT, p: Lit, from: ?*Clause) void {
        if (self.litValue(p) != types.l_Undef) {
            unreachable;
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

    fn propagate(self: *MiniSAT) *Clause {
        // TODO:
        _ = self;
        unreachable;
    }

    inline fn reason(self: MiniSAT, x: Var) ?*Clause {
        return self.vardata.get(x).?.reason;
    }

    inline fn level(self: MiniSAT, x: Var) usize {
        return self.vardata.get(x).?.level;
    }

    fn insertVarOrder(self: MiniSAT, v: Var) void {
        if (!self.inOrderHeap(v) and self.decision.get(v)) {
            self.order_heap.insert(v);
        }
    }

    fn inOrderHeap(self: MiniSAT, v: Var) bool {
        for (self.order_heap) |x| {
            if (x == v) {
                return true;
            }
        }
        return false;
    }

    inline fn varDecayActivity(self: *MiniSAT) void {
        self.var_inc *= 1 / self.var_decay;
    }

    fn varBumpActivity(self: *MiniSAT, v: Var) void {
        const act: *f64 = self.activity.getPtr(v).?;
        act.* += self.var_inc;
        if (act.* > 1e100) {
            for (0..self.newVar()) |i| {
                self.activity.put(@intCast(i), act.* * 1e-100) catch unreachable;
            }
            self.var_inc *= 1e-100;
        }
        if (self.inOrderHeap(v)) {
            self.order_heap.update(v, v);
        }
    }

    inline fn claDecayActivity(self: *MiniSAT) void {
        self.cla_inc *= 1 / self.clause_decay;
    }

    fn claBumpActivity(self: *MiniSAT, c: *Clause) void {
        const act: *f32 = c.activityPtr().?;
        act.* += self.cla_inc;
        if (c.activity > 1e20) {
            for (0..self.learnts.len) |i| {
                const act_i: *f32 = self.learnts.items[i].activityPtr().?;
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
        const val_0 = self.litValue(var_0);
        return val_0 == types.l_True and self.reason(var_0) == c;
    }

    inline fn newDecisionLevel(self: *MiniSAT) void {
        self.trail_lim.append(self.trail.len);
    }

    inline fn decisionLevel(self: MiniSAT) usize {
        return self.trail_lim.items.len;
    }

    inline fn abstractLevel(self: MiniSAT, x: Var) u32 {
        return 1 << (self.level(x) & 31);
    }

    inline fn litValue(self: MiniSAT, l: Lit) Lbool {
        return self.assigns.get(l.variable()).? ^ @as(i32, l.sign());
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
        return self.trail.len;
    }

    inline fn nVars(self: MiniSAT) usize {
        return self.next_var;
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
