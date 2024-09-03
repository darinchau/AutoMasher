# Frontend

The frontend web demo should be a one-page easy-to-use interface that allows users to put in a song and get our mashups.

The user should be able to input a song by either uploading an audio file or by providing a link to a song on YouTube. The system should be able to show a thumbnail and an audio preview of the song that the user has inputted.

After the user has inputted a song. The user should then be able to click a button to generate a mashup. The system should then display a personalized link to download the mashup.

The user should also be able to access a dropdown menu that gives them more control to the mashup config.

# Backend

The backend should be a REST API that takes in a song, and a mashup config. The API should then return a link to download the mashup.

# Submitting chord and beat results
The user should label a song with a starting point of their mashup. The algorithm will pick an appropriate starting point as follows:
- If the user's input is within 20 seconds of the end of the music, we prompt the user to re-enter the starting point.
- The appropriate starting point should have at least 8 more downbeats. These 8 downbeats should be consistent i.e. the range of the differences should be within 0.9x to 1.1x of the average difference.
- We search for downbeats within 3 seconds of the user's input. If:
- - There is only one downbeat within +- 3 seconds of the user's input, we use that as the starting point.
- - There are multiple downbeats within +- 3 seconds of the user's input, we try to run a tiebreaker algorithm to determine the correct downbeat.
- - - At each downbeat, perform the cadence detection algorithm. The downbeat with the highest cadence score is the correct downbeat.
- - - If tiebreaker fails, pick the closest one that satisfies the basic requirements.
- - There are no downbeats within +- 3 seconds of the user's input, we will run a music detection algorithm
- - - If there is no music, we forward-shift the starting point to the first instance with music, and rerun this 3 second algorithm.
- - - If there is music, we will apply an extrapolation algorithm to extend the beats to the user's input. After extending the beats, we will rerun the 3 second algorithm.
