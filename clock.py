"""
Author: Dr Zhibin Liao
Organisation: School of Computer Science and Mathematical Sciences, University of Adelaide
Date: 03-Apr-2025
Description: This Python script is a wrapper of the game clock. This makes the game runnable without showing a screen.

The script is a part of Assignment 2 made for the course COMP SCI 3007/7059/7659 Artificial Intelligence for the year
of 2025. Public distribution of this source code is strictly forbidden.
"""
import pygame


class ClockWrapper:
    def __init__(self, show_screen=False, frame_rate=30):
        self.show_screen = show_screen
        self.frame_rate = frame_rate

        if self.show_screen:
            self.clock = pygame.time.Clock()
        else:
            self.clock_counter = 0

    def current_time(self):
        if self.show_screen:
            return pygame.time.get_ticks()
        else:
            return self.clock_counter

    def tick(self):
        if self.show_screen:
            self.clock.tick(self.frame_rate)  # frame rate
        else:
            self.clock_counter += 1000 / self.frame_rate