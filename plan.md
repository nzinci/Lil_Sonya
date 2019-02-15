1. This article will be used as base: "This Time with Feeling:
Learning Expressive Musical Performance"
2. Data preparation. For convenience and consistency of the data for our purposes of learning, we
  will use classical music in solo interpretation for the piano, that is, data that which is 
 captured in the MIDI representation. Therefore, it is necessary to figure out how to process
 this  data for training and where to get it at all.
Find a free library with them and pick up data from about the same category, that is, as in music,
 the model should try to withstand the style when creating (17-19 feb) 

3. My network will operates on a one-hot encoding.Will be used LSTM-RNN model. First, try the simplest 
version of RNN (baseline RNN trained on 15
 -second clips). If it somehow will work, the same thing, but with the addition of a pedal (i.e.
 baseline with pedaled notes extended). If a miracle happened and this model also somehow
 produces  sounds, then I try to arrange an ultra-mega combination with a model of this type:
 baseline +  pedal + 30-second clips (The first launch with the simplest model, I be
 closer to the  second of March.The first launch with the simplest model I hope happens closer to
 the 10 mar. Then  for each model and debugging in case of a shift in time - plus one week.
If the second model gives about the same result, leave the idea with it and try to work with the 
first simplest model and optimize it (perhaps due to more correct work with the data) 

4. Every week, if necessary, there will be an adjustment to the schedule. Also in the plan will
be  added more accurate details of the architecture and the whole project.
