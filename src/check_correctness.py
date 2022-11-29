from generate_csv import build_coordinate_mapping_table as Table


def coord_mapping_table():
	"""
	Check coordinate mapping table.
	"""

	dic = Table()

	for x in range(-200, 200 + 1):
		for y in range(-200, 200 + 1):
			nx = (abs(x) // 50 + (1 if abs(x) % 50 != 0 else 0)) * (1 if x > 0 else -1) * 50
			ny = (abs(y) // 50 + (1 if abs(y) % 50 != 0 else 0)) * (1 if y > 0 else -1) * 50
			if nx == 0: nx = 50
			if ny == 0: ny = 50
			print(x, y, dic[(nx, ny)])
