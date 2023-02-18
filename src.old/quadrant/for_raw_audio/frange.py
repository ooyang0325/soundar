from tqdm import tqdm, trange

def frange(*args):

	# Better Progress Bar

	if len(args) == 1:
		end = int(args[0])
		start, step = 0, 1
	elif len(args) == 2:
		start, end = map(int, args)
		step = 1
	else:
		start, end, step = map(int, args)
	
	return tqdm(range(start, end, step), ascii=" ▖▘▝▗▚▞█")

