const std = @import("std");

pub fn index_of(comptime T: type, slice: []const T, value: T) ?usize {
    for (slice, 0..) |element, index| {
        if (std.meta.eql(value, element)) return index;
    } else return null;
}
