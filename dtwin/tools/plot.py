#!/bin/python

import matplotlib.pyplot as plt
import pandas as pd
import math
import sys
import io

df = pd.read_csv(sys.argv[1], names=['Frame Number', 'Value'])

plt.figure(figsize=(10, 6))
plt.plot([math.log(x+1) for x in df['Frame Number']], df['Value'])
plt.title('Frame Metadata: Value vs Frame Number')
plt.xlabel('Frame Number')
plt.ylabel('Emergence Index')
plt.show()
