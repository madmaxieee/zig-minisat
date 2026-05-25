const std = @import("std");

pub fn IntMap(comptime Key: type, comptime Value: type) type {
    return struct {
        const Self = @This();
        items: std.ArrayList(Value),
        pub fn init(allocator: std.mem.Allocator) Self {
            return Self{ .items = std.ArrayList(Value){} };
        }
        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            self.items.deinit(allocator);
        }
        pub fn getIndex(key: Key) usize {
            return @intCast(key);
        }
        pub fn hasKey(self: Self, key: Key) bool {
            return Self.getIndex(key) < self.items.items.len;
        }
        pub fn get(self: *Self, key: Key) ?Value {
            const index = Self.getIndex(key);
            if (index >= self.items.items.len) {
                return null;
            }
            return self.items.items[index];
        }
        pub fn resize(self: *Self, allocator: std.mem.Allocator, capacity: usize) !void {
            try self.items.resize(allocator, capacity);
        }
        pub fn set(self: *Self, allocator: std.mem.Allocator, key: Key, value: Value) !void {
            const index = Self.getIndex(key);
            if (index >= self.items.items.len) {
                try self.items.resize(allocator, index + 1);
            }
            self.items.items[index] = value;
        }
        pub fn clear(self: *Self) void {
            self.items.clearRetainingCapacity();
        }
    };
}

test "IntMap" {
    const testing = std.testing;
    const test_allocator = testing.allocator;
    var map = IntMap(u64, u8){ .items = std.ArrayList(u8){} };
    defer map.deinit(test_allocator);
    try map.set(test_allocator, 1, 2);
    try map.set(test_allocator, 3, 4);
    try testing.expect(map.get(1) == 2);
    try testing.expect(map.get(3) == 4);
    try testing.expect(map.get(5) == null);
}

pub fn IntSet(comptime Key: type) type {
    return struct {
        const Self = @This();
        items: std.ArrayList(bool),
        map: IntMap(Key, bool),
        pub fn init(allocator: std.mem.Allocator) Self {
            return .{
                .items = std.ArrayList(bool){},
                .map = IntMap(Key, bool).init(allocator),
            };
        }
        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            self.map.deinit(allocator);
            self.items.deinit(allocator);
        }
        pub fn getIndex(key: Key) usize {
            return @intCast(key);
        }
        pub fn has(self: Self, key: Key) bool {
            return self.map.hasKey(key);
        }
    };
}
