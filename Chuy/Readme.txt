Data Sheet for Project: Intelligent RF Sensing for Falls and Health Prediction – INSHEP (EP/R041679/1)
James Watt School of Engineering, University of Glasgow
Dr Syed Aziz Shah, Dr Julien Le Kernec, Dr Francesco Fioranelli (PI)
{syed.a.shah, julien.lekernec, francesco.fioranelli}@glasgow.ac.uk
This document is for the data obtained in March 2017 (University of Glasgow), December 2017 (University of
Glasgow), June 2017 (University of Glasgow), July 2018 (University of Glasgow), February 2019 (North Glasgow
Housing Association facilities), and March 2019 (Age UK West Cumbria centre).
What did people do?
They were asked to perform two or three repetitions of five different activities, which were walking back and
forth, sitting down on a chair, standing up, bending to pick up an object, and drinking from a cup or glass. We
asked people to perform each activity separately one from another in “snapshots.
In some cases, an additional activity, a simulated frontal fall, was also collected. This was only possible in
laboratory-controlled conditions and only for some subjects for safety reasons.
What is the aim?
The ultimate aim is to create a system (device plus algorithms) that is capable of monitoring the activity levels
and patterns of people, to
-detect critical events such as a fall;
-learn the usual activity level so that changes can be promptly detected and in the case discussed with the
person and where needed health professionals (is the person walking less than usual? Is the person more
sedentary and less active? Is the person showing signs of more random behaviour? And so on).
We propose radar as a sensor because it is contactless (no need to wear or touch any sensor) and more privacy
compliant than cameras (no plain pictures or videos of faces or private spaces are collected).
Data collection.
The data was collected using an off-the-shelf FMCW radar (by Ancortek) operating at C-band (5.8 GHz) with
bandwidth 400 MHz and chirp duration 1ms, delivering an output power of approximately +18 dBm as shown
in Fig. 1. The radar is connected to transmitting and receiving Yagi antennas with a gain of about +17dB and is
capable of recording micro-Doppler signatures of the people moving within the area of interest.

Fig. 1 – Typical radar configuration with transmitting and receiving antennas (the radar is the blue device on the table,
and the two antennas are the white cylinders on the two support tripods)

File format.
The data have been organised in separate folders with .dat files for each data collection, with details provided
in the remainder of this document.
The data files have been named with this generic approach KPXXAYYRZ so that
 the digits K 1 2 3 4 5 and 6 at the beginning indicates the activities walking, sitting down, stand up, pick up
an object, drink water, and fall respectively;
 the characters XX indicate the subject (individual person) having ID XX (01, 02, etc…);
1

 the characters YY indicate the activity being performed such as A01, A02, A03, A04, A05, and A06;
 the character Z indicates the repetition of the activity such as R1, R2, etc.
Some information about the subjects (age, height, gender, dominant hand) is also reported in this document
as metadata. Note also that in some cases not all the information were available, and this has been replaced
by n/a for some subjects.
When imported into MATLAB (or equivalent software), each file is seen as a long 1D complex array (or table).
The first 4 elements include in this order the carrier frequency (5.8 GHz), the duration of the chirp (1 ms), the
number of samples per recorded beat-note signal (128 samples), and the bandwidth of the chirp (400 MHz);
the following elements are the complex samples of the sequence of recorded beat-notes one after the other.

1 - December 2017 Datasets





360 DAT Files
6 activities including walking, sitting down, standing up, pick up an object, drink water and fall
20 volunteers participated
3 repetitions
Location - University of Glasgow laboratory room

Subject ID

Age

Height [cm]

P36
P37
P38
P39
P40
P41
P42
P43
P44
P45
P46
P47
P48
P50
P51
P52
P53
P54
P55
P56

27
27
28
23
22
23
22
25
27
25
31
27
34
24
26
26
21
21
23
32

182
176
182
182
183
185
180
181
167
173
167
180
172
182
178
170
180
180
188
170

Dominant
Hand
Right Hand
Right Hand
Right Hand
Right Hand
Right Hand
n/a
Right Hand
Right Hand
Right Hand
Right Hand
Right Hand
Right Hand
Right Hand
Right Hand
Right Hand
Right Hand
n/a
Right Hand
Right Hand
Right Hand

Gender
Male
Male
Male
Male
Male
Male
Male
Male
Male
Male
Male
Male
Male
Male
Male
Male
Male
Male
Male
Male
2

2 - March 2017 Datasets





48 DAT Files
6 activities including walking, sitting down, standing up, pick up an object, drink water and fall
4 volunteers participated
2 repetitions
Subject ID Age Height [cm] Dominant Hand Gender
P03
23
180
Right Hand
Male
P10
23
182
Right Hand
Male
P11
23
182
Right Hand
Male
P12
31
170
Right Hand
Male

3 - June 2017 Datasets





162 DAT Files
6 activities including walking, sitting down, standing up, pick up an object, drink water and fall
9 volunteers participated
3 repetitions
Subject
ID
P14
P28
P29
P30
P31
P32
P33
P34
P35

Age
n/a
27
27
23
23
26
24
36

Height
[cm]
n/a
180
176
180
149
173
173
176
175

Dominant
Hand
n/a
Left hand
Right hand
Right hand
Right hand
Right hand
Right hand
Left hand
Right hand

Gender
Female
Male
Male
Male
Female
Male
Male
Male
Male

3

4 - July 2018 Dataset





288 DAT Files
6 activities including walking, sitting down, standing up, pick up an object, drink water and fall
16 volunteers participated
3 repetitions
Location - University of Glasgow common room

Subject ID

Age

Height [cm]

P57
P58
P59
P60
P61
P62
P63
P64
P65
P66
P67
P68
P69
P70
P71
P72

32
25
32
25
27
26
27
28
23
26
27
25
36
26
24
28

170
168
168
170
173
173
178
177
180
180
165
180
182
180
178
168

Dominant
Hand
Right Hand
Right Hand
Left Hand
Right Hand
Right Hand
Right Hand
Right Hand
Right Hand
Right Hand
Right Hand
Right Hand
Right Hand
Right Hand
Right Hand
Right Hand
Right Hand

Gender
Male
Female
Male
Male
Male
Male
Male
Male
Male
Male
Male
Male
Male
Male
Male
Male

4

5 -February 2019 Dataset UoG Dataset





306 DAT Files
6 activities including walking, sitting down, standing up, pick up an object, drink water and fall
17 volunteers participated
3 repetitions
Location - University of Glasgow laboratory room

Subject ID

Age

Height [cm]

P01
P02
P03
P04
P05
P06
P07
P08
P09
P10
P11
P12
P13
P14
P15
P16
P17

25
37
32
36
31
44
34
33
30
27
25
25
31
32
27
25
25

180
182
183
170
170
177
165
170
167
173
161
182
179
168
181
180
180

Dominant
Hand
Right Hand
Right Hand
Right Hand
Right Hand
Right Hand
Right Hand
Right Hand
Right Hand
Right Hand
Right Hand
Right Hand
Left Hand
Right Hand
Right Hand
Right Hand
Right Hand
Right Hand

Gender
Male
Male
Male
Male
Male
Male
Female
Male
Male
Male
Female
Male
Male
Male
Male
Male
Male

5

6 - February 2019 Dataset NG Homes Dataset





301 DAT Files
5 activities including walking, sitting down, standing up, pick up an object and drink water
20 volunteers participated
3 repetitions (note that for P21 and P23 one repetition of A01 walking could not be recorded; for P08 also
three extra repetitions of activity 06 falling were recorded)
Location – Glasgow NG Homes Room 1

Subject ID

Age

Height [cm]

P08
P18
P19
P20
P21
P22
P23
P24
P25

33
65
82
78
66
33
50
56
25

170
164.5
170.6
170.6
152.4
161.5
155.5
152.4
155.4

Dominant
Hand
Right Hand
Left Hand
Right Hand
Right Hand
Right Hand
Right Hand
Right Hand
Right Hand
Right Hand

Gender
Male
Male
Male
Male
Male
Female
Male
Male
Female

Location – Glasgow NG Homes Room 2

Subject ID

Age

Height [cm]

P26
P27
P28
P29
P30
P31
P32

88
63
79
68
65
24
84

n/a
176.7
176.7
n/a
178.3
n/a
n/a

Dominant
Hand
Right Hand
Right Hand
Left Hand
Right Hand
Right Hand
Right Hand
Right Hand

Gender
Male
Male
Male
Female
Male
Male
Male
6

Location – Glasgow NG Homes Room 3

Subject ID

Age

Height [cm]

P33
P34
P35
P36

79
60
64
70

n/a
n/a
n/a
n/a

Dominant
Hand
Right Hand
Left Hand
Right Hand
Right Hand

Gender
Female
Female
Female
Female

7 - March 2019 Dataset West Cumbria Dataset





289 DAT Files
5 activities including walking, sitting down, standing up, pick up an object, drink water and fall
20 volunteers participated
3 repetitions (with the exception of P42 for which only limited data were collected)
Location – Age Uk West Cumbria Room 1

Subject ID
P37
P38
P39
P40
P41
P42
P43
P44
P45
P46
P47
P48
P49

Age
75
74
52
48
84
85
67
45
78
67
98
47
57

Height [cm]
n/a
n/a
n/a
n/a
n/a
n/a
n/a
n/a
n/a
n/a
n/a
n/a
n/a

Dominant Hand
Right Hand
Right Hand
Left hand
Right Hand
Right Hand
Right Hand
Right Hand
Right Hand
Right Hand
Right Hand
Right Hand
Right Hand
Right Hand

Gender
Female
Female
Male
Female
Female
Male
Male
Female
Female
Female
Female
Female
Male
7

Location – Age Uk West Cumbria Room 1

Subject ID
P50
P51
P52
P53
P54
P55
P56

Age
71
50
49
84
69
57
25

Height [cm]
n/a
n/a
n/a
n/a
n/a
n/a
n/a

Dominant Hand
Right Hand
Right Hand
Right Hand
Right Hand
Right Hand
Right Hand
Right Hand

Gender
Female
Female
Female
Male
Female
Female
Female

8

