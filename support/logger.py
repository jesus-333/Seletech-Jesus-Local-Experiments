#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 14:52:26 2022

@author: Alberto Zancanaro (jesus)
@organization: University of Padua
"""

#%% Imports

from datetime import datetime

#%% Logger class

class AnotherLogger():
    
    def __init__(self, file_path, level = 0, clean_file = True, extra_empty_line = True):
        self.current_level = 0
        
        self.last_txt = ""
        
        self.file_path = file_path
        
        self.extra_empty_line = extra_empty_line
        
        # Create and clean the file
        if clean_file: open(file_path, 'w').close() 
    
    def write_message(self, message, level, message_type = "LOG"):
        if level > self.current_level: # Write only if the message is important enough
            self.last_txt = self.get_date_time() + "\t" + message_type + "\n\t" + message + "\n"
            if self.extra_empty_line: self.last_txt += "\n"
            
            with open(self.file_path, 'a') as fd:
                fd.write(self.last_txt)
                
    def get_date_time(self):
        return datetime.now().strftime("%d/%m/%Y %H:%M:%S.%f")
    
    def setLevel(self, level): self.current_level = level
        
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -          
    
    def debug(self, message): self.write_message(message, 10, 'DEBUG')
    def info(self, message): self.write_message(message, 20, 'INFO')
    def warning(self, message): self.write_message(message, 30, 'WARNING')
    def error(self, message): self.write_message(message, 40, 'ERROR')
    def critical(self, message): self.write_message(message, 50, 'CRITICAL')
        
            
