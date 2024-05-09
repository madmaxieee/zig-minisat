const std = @import("std");

const types = @import("types.zig");
const Var = types.Variable;
const Lbool = types.LiftedBool;

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

pub const MiniSAT = struct {
    model: []Lbool,

    pub fn solver(self: *MiniSAT) Solver {
        return Solver.init(self);
    }

    pub fn newVar(self: *MiniSAT, upol: Lbool, dvar: bool) Var {
        _ = upol;
        _ = dvar;
        _ = self;
        return 1;
    }
};
