**TextGrids**
- min and max/start and end times of textgrid wrt audio file
- number of tiers
- data of each tier: intervals (have duration) or points (instaneous) they contain 

<br/>
<br/>

**Opening TextGrids and getting info**

<u>Opening TextGrid</u>

```
from os.path import join

from praatio import textgrid

inputFN = join('..', 'examples', 'files', 'mary.TextGrid')

tg = textgrid.openTextgrid(inputFN, includeEmptyIntervals=False)
```

<br/>

<u>Getting a Tier</u>

- TextGrid tiers are stored in tierDict
- TextGrid tier names are stored in tierNames -> tg.tierNames

```
firstTier = tg.getTier(tg.tierNames[0])
```

<br/>

<u>Accessing intervals or points within a Tier</u>

- intervals or points are stored as entries within a tier

For a <em>pointTier</em>, the entries look like

```
[(time, labe1), (time, label), ...]
```

For a <em>intervalTier</em>, the entries look like

```
[(start, end, label), (start, end, label), ...]
```

Getting the labels and duration of each label from the entry
- open textgrid, get target tier, and forloop through the entries

```
wordTier = tg.getTier('word')

# I just want the labels from the entries
labelList = [entry.label for entry in wordTier.entries]
print(labelList)

# Get the duration of each interval
# (in this example, an interval is a word, so this outputs word duration)
durationList = []
for start, stop, _ in wordTier.entries:
    durationList.append(stop - start)

print(durationList)
```

<br/>

```
# Print out each interval on a separate line
from os.path import join
from praatio import textgrid

inputFN = join('..', 'examples', 'files', 'mary.TextGrid')
tg = textgrid.openTextgrid(inputFN, includeEmptyIntervals=False)
tier = tg.getTier('word') # Get tiers by their names
for start, stop, label in tier.entries:
    print("From:%f, To:%f, %s" % (start, stop, label))
```

<br/>
<br/>

**Cropping TextGrids**


```
tg.crop(startTime, endTime, mode, rebaseToZero)


Args:
    cropStart: The start time of the crop interval
    cropEnd: The stop time of the crop interval
    mode: Determines the crop behavior
        - 'strict', only intervals wholly contained by the crop
            interval will be kept
        - 'lax', partially contained intervals will be kept
        - 'truncated', partially contained intervals will be
            truncated to fit within the crop region.
    rebaseToZero: if True, the cropped textgrid timestamps will be
        subtracted by the cropStart; if False, timestamps will not
        be changed

Returns:
    the modified version of the current textgrid
```

- startTime, endTime: start and end times that define the crop region
- mode
- rebaseToZero: if True, the entry time values will be subtracted by startTime




```
# Crop takes four arguments
# If mode is 'truncated', all intervals contained within the crop region will appear in the
# returned TG--however, intervals that span the crop region will be truncated to fit within
# the crop region
# If rebaseToZero is True, the times in the textgrid are recalibrated with the start of
# the crop region being 0.0s
```

<br/>

```
# If rebaseToZero is False, the values in the cropped textgrid will be what they were in the
# original textgrid
```

<br/>

```
# If mode is 'strict', only wholly contained intervals will be included in the output.
# Compare this with the previous result
```

<br/>

```
# If mode is 'lax', partially contained intervals will be wholly contained in the outpu.
# Compare this with the previous result
```

<br/>
<br/>

**Cropping Tiers**

```
tg.crop(startTime, endTime, mode, rebaseToZero)
```

<br/>
<br/>

**Other Useful Things**

- ```splitAudioOnTier()```
- other ready-to-use functions in ```/praatio/praatio_scripts.py```