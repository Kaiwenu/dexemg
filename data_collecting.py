''' 
Gathering data and training labels
Data is saved in foo.csv in working directory
'''
import pygame
import multiprocessing
import numpy as np
import time
import save_hdf5

# import commonas c
# from paddle import Paddle
from pyomyo import Myo, emg_mode

# ------------ Myo Setup ---------------
q = multiprocessing.Queue()

def worker(q, MODE):
	m = Myo(mode=MODE)
	m.connect()

	def add_to_queue(emg, movement):
		q.put(emg)

	m.add_emg_handler(add_to_queue)

	 # Orange logo and bar LEDs
	m.set_leds([128, 128, 0], [128, 128, 0])
	# Vibrate to know we connected okay
	m.vibrate(1)

	"""worker function"""
	while True:
		m.run()
	print("Worker Stopped")

#Plot sEMG signal
last_vals = None
def plot(scr, vals):
	w = WIN_X
	h = WIN_Y
	DRAW_LINES = True

	global last_vals
	if last_vals is None:
		last_vals = vals
		return

	D = 5
	scr.scroll(-D)
	scr.fill((0, 0, 0), (w - D, 0, w, h))
	for i, (u, v) in enumerate(zip(last_vals, vals)):
		if DRAW_LINES:
			pygame.draw.line(scr, (0, 255, 0),
							 (w - D, int(h/9 * (i+1 - u))),
							 (w, int(h/9 * (i+1 - v))))
			pygame.draw.line(scr, (255, 255, 255),
							 (w - D, int(h/9 * (i+1))),
							 (w, int(h/9 * (i+1))))
		else:
			c = int(255 * max(0, min(1, v)))
			scr.fill((c, c, c), (w - D, i * h / 8, D, (i + 1) * h / 8 - i * h / 8))

	pygame.display.flip()
	last_vals = vals


# -------- Main Program Loop -----------
if __name__ == "__main__":
	score = 0
	lives = 3
	MOVE_SPEED = 10

	# Experiment vars
	MODE = emg_mode.PREPROCESSED
	TIMER = True

	start_time = time.time()
	start_time_ns = time.perf_counter_ns() 
	p = multiprocessing.Process(target=worker, args=(q,MODE,))
	p.start()

	# PyGame setup 
	pygame.init()

	# Open a new window
	SCALE = int(2)
	WIN_X = 400 * SCALE
	WIN_Y = 300 * SCALE
	size = (WIN_X, WIN_Y)
	screen = pygame.display.set_mode(size)
	pygame.display.set_caption("sEMG Signal")
	 
	# This will be a list that will contain all the sprites we intend to use in our game.
	all_sprites_list = pygame.sprite.Group()
	 

	# The loop will carry on until the user exit the game (e.g. clicks the close button).
	carryOn = True
	 
	# The clock will be used to control how fast the screen updates
	clock = pygame.time.Clock()

	emg_samples = []
	t_samples = []

	while carryOn:
		# --- Main event loop
		for event in pygame.event.get(): # User did something
			if event.type == pygame.QUIT: # If user clicked close
				  carryOn = False # Flag that we are done so we exit this loop

		while not(q.empty()):
			d = list(q.get())
			emg_samples.append(d)
			t_samples.append(time.time())
			plot(screen, [e / 500. for e in d])
			
			
		if (TIMER):
			'''
			Stop recording data if we have reached the timelimit
			'''
			time_elapsed = time.time() - start_time
			if (time_elapsed > 100):
				print(f"Timer Activated: {time_elapsed}")
				carryOn = False
			
		if carryOn == False:		
			# Handle data

			pygame.quit()
			p.terminate()
			p.join()

	# Once we have exited the main program loop we can stop the game engine:
	pygame.quit()

	out = save_hdf5.save_emg2pose_hdf5(
				out_path="my_recording_right.hdf5",
				emg_samples=emg_samples,
				t_samples=t_samples,
				user="kaich",
				side="right",
				split="test",
				stage="OneHandedFreeStyle",
			)
	print("Saved:", out)