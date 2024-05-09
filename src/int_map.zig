const std = @import("std");

pub fn IntMap(comptime Key: type, comptime Value: type) type {
    return struct {
        map: std.ArrayList(Value),
        const Self = @This();
        pub fn init(allocator: std.mem.Allocator) Self {
            return .{ .map = std.ArrayList(Value).init(allocator) };
        }
        pub fn deinit(self: *Self) void {
            self.map.deinit();
        }
        pub fn getIndex(key: Key) usize {
            return @intCast(key);
        }
        pub fn hasKey(self: Self, key: Key) bool {
            return Self.getIndex(key) < self.map.items.len;
        }
        pub fn get(self: *Self, key: Key) ?Value {
            const index = Self.getIndex(key);
            if (index >= self.map.items.len) {
                return null;
            }
            return self.map.items[index];
        }
        pub fn resize(self: *Self, capacity: usize) !void {
            try self.map.resize(capacity);
        }
        pub fn set(self: *Self, key: Key, value: Value) !void {
            const index = Self.getIndex(key);
            if (index >= self.map.items.len) {
                try self.map.resize(index + 1);
            }
            self.map.items[index] = value;
        }
        pub fn clear(self: *Self) void {
            self.map.clearRetainingCapacity();
        }
    };
}

test "IntMap" {
    const testing = std.testing;
    const test_allocator = testing.allocator;
    var map = IntMap(u64, u8){ .map = std.ArrayList(u8).init(test_allocator) };
    defer map.deinit();
    try map.set(1, 2);
    try map.set(3, 4);
    try testing.expect(map.get(1) == 2);
    try testing.expect(map.get(3) == 4);
}
