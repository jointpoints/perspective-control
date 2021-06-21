__author__ = 'Andrei Eliseev'





def start_progress(verbosity, label, label_capacity):
	if verbosity:
		print('\n' + label + ' ' * (label_capacity - len(label)), end='', flush=True)
	return





def update_progress(verbosity, share):
	if verbosity:
		print('#' * share, end='', flush=True)
	return
