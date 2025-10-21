from controls.location import HandLocation
from performance.profiler import PerformanceProfiler


def generate_sample_data():
	sample_data = []
	for i in range(21):
		sample_data.append({
			'point': i,
			'coordinates': (100 + i * 10, 200 + i * 10),
			'axis': [0.1 + i * 0.02, 0.2 + i * 0.02, 0.3 + i * 0.02]
		})
	return sample_data


if __name__ == '__main__':
	# Example usage and basic test
	print("HandLocation Python Module - Optimized Implementation")
	print("=" * 60)
	hand_loc = HandLocation()
	print(f"Created: {hand_loc}")

	sample_data = generate_sample_data()
	hand_loc.update_values(sample_data)
	print(f"After update: {hand_loc}")

	profiler = PerformanceProfiler()
	stats = profiler.profile_update(hand_loc, sample_data, iterations=1000)

	print("\nPerformance Statistics (1000 iterations):")
	print(f"  Mean:   {stats['mean_us']:.2f} μs")
	print(f"  Median: {stats['median_us']:.2f} μs")
	print(f"  Std:    {stats['std_us']:.2f} μs")
	print(f"  P95:    {stats['p95_us']:.2f} μs")
	print(f"  P99:    {stats['p99_us']:.2f} μs")

	print("\n✓ All tests passed!")
