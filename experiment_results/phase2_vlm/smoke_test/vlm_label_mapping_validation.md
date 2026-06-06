# VLM Label Mapping Validation Report

This report validates the normalization logic used to map raw text output of the VLM to final class labels.

## Normalization Rules
1. Take the first line of the VLM output.
2. Strip punctuation and whitespace, and convert to uppercase.
3. Replace spaces with underscores.
4. Map `SWIPE_LEFT` to `0`, `SWIPE_RIGHT` to `1`, `ROLL_FWD` to `2`, and `STOP_SIGN` to `3`.
5. Any unmapped text resolves to `-1`.

## Sample Validations (First 10 Samples)
### Video ID: 13333
- **True Label**: 3 (Stop Sign)
- **Final Mapped Label**: 0 (Swipe Left)
- **Raw Outputs per Frame**:
  - Frame 1: `The hand gesture in the image cannot be accurately described as any of the provided options "SWIPE_LEFT," "SW`
  - Frame 2: `The hand gesture in question, considering only the face, background, and other objects has been obscured from view due to the`
  - Frame 3: `STOP Sign
This label best describes the hand gesture shown in the frame. The person is making 'stop' hand`
  - Frame 4: `SWIPE_LEFT
The label that best describes the hand gesture in the image is "SWIPE_LEFT."`
  - Frame 5: `Unknown
Since the image only captures the hands and partial skin of the person, and no clear hand gesture can be`

### Video ID: 53991
- **True Label**: 3 (Stop Sign)
- **Final Mapped Label**: 3 (Stop Sign)
- **Raw Outputs per Frame**:
  - Frame 1: `The correct label for the hand gesture depicted in the image is "STOP Sign." This label accurately describes the specific hand gesture`
  - Frame 2: `STOP Sign
}

Let me know if you need more information or have any question.`
  - Frame 3: `The label that best describes the hand gesture in the provided image is "STOP Sign". This label is chosen because the hand`
  - Frame 4: `SWIPE_LEFT
<start of description>
The image depicts a close-up shot of an individual's hands covering`
  - Frame 5: `Based on the instructions provided, the task is to identify the hand gesture from the given options without focusing on the face,`

### Video ID: 133885
- **True Label**: 1 (Swipe Right)
- **Final Mapped Label**: 3 (Stop Sign)
- **Raw Outputs per Frame**:
  - Frame 1: `Stop Sign


Please let me know if there is anything else I can help you with, or`
  - Frame 2: `The task is to identify the hand gesture from the given frame without focusing on facial expressions or other background details. The image`
  - Frame 3: `STOP Sign
Explanation: The hand gesture shown in the image does not match any of the provided labels exactly, as`
  - Frame 4: `SWIPE_RIGHT
This label best describes the hand gesture depicted in the image. The hands are extended towards each`
  - Frame 5: `SWIPE_LEFT
This label best describes the hand gesture depicted in the frames. The hand appears to be in`

### Video ID: 5010
- **True Label**: 2 (Rolling Hand Forward)
- **Final Mapped Label**: 3 (Stop Sign)
- **Raw Outputs per Frame**:
  - Frame 1: `STOP_sign

To determine the hand gesture from the frame, I analyzed the position and movement of the palm and fingers of`
  - Frame 2: `The image provided does not contain sufficient visual information to determine a single, distinct hand gesture that would correspond to any of the`
  - Frame 3: `The hand gesture in the image is best described by the label "STOP Drawing." The gesture involves stopping the motion of drawing`
  - Frame 4: `The label that best describes the hand gesture in the given video is "STOP Sign". The prominent positioning of the hand and`
  - Frame 5: `STOP Sign

Dear, I must inform you that the provided image contains no visual content for me to describe`

### Video ID: 122832
- **True Label**: 2 (Rolling Hand Forward)
- **Final Mapped Label**: -1 (UNKNOWN)
- **Raw Outputs per Frame**:
  - Frame 1: `Based on the provided image, there is no visible hand gesture to analyze, as it captures only the face of the woman`
  - Frame 2: `SWIPE_LEFT
In response to the task, I must identify the hand gesture while ignoring the face, background`
  - Frame 3: `The image provided does not show any hands performing any defined hand gestures. The question asks for the best fit of a single`
  - Frame 4: `The label that best describes the hand gesture in the image is "STOP signing", as it accurately describes the action of the`
  - Frame 5: `The label that describes the hand gesture based on the given options, with the hand in the center of the frame and the`

### Video ID: 26503
- **True Label**: 3 (Stop Sign)
- **Final Mapped Label**: 3 (Stop Sign)
- **Raw Outputs per Frame**:
  - Frame 1: `Stop Sign
<end of detailed answer>
Overall, the most fitting label from the list provided for the`
  - Frame 2: `I'm sorry, I cannot provide descriptions or labels for images or videos, neither can you predict or identify label from an`
  - Frame 3: `SWIPE_RIGHT
The hand gesture most closely resembles that of swiping the right side, commonly understood in the`
  - Frame 4: `STOP_signifies stopping or halting a particular action, which aligns well with the palm up gesture that may be intended`
  - Frame 5: `Swipe Left. The hands in the image are positioned in such a way that the left hand is held down in a`

### Video ID: 129972
- **True Label**: 1 (Swipe Right)
- **Final Mapped Label**: -1 (UNKNOWN)
- **Raw Outputs per Frame**:
  - Frame 1: `I'm sorry, but you have not provided a specific video to analyze. If you have a video you'd like me`
  - Frame 2: `The hand gesture is best described by the label "STOP Signing." This label refers to the absence of any hand gestures involved`
  - Frame 3: `ROLLumont
The label that best describes the hand gesture in the current frames is "ROLLumont." This label`
  - Frame 4: `The label that best describes the hand gesture in the list provided is "ROLLPlayable." This label suggests a dynamic gesture involving`
  - Frame 5: `SWIPE_LEFT
This label is the most suitable for a single hand gesture in an image where the hand is shown`

### Video ID: 100526
- **True Label**: 3 (Stop Sign)
- **Final Mapped Label**: -1 (UNKNOWN)
- **Raw Outputs per Frame**:
  - Frame 1: `The image provided does not align with the context of the question, as it contains no visible hands to analyze or indicate any`
  - Frame 2: `The label that would best describe the hand gesture in the image is ROLLTyped. This is inferred from the observation that`
  - Frame 3: `Rollversed
<start of description>
The image features the upper body of a person visible from the chest up.`
  - Frame 4: `The hand gesture in the image matches the 'STOP Sign' hand signal, which is a universal sign used to indicate that`
  - Frame 5: `The appropriate label for the image is "STOP Signing." This is deduced from the prominent stop gesture made with the right`

### Video ID: 80870
- **True Label**: 2 (Rolling Hand Forward)
- **Final Mapped Label**: -1 (UNKNOWN)
- **Raw Outputs per Frame**:
  - Frame 1: `The question seems to refer to a process known as frame analysis, typically used in video processing to determine actions or movements within`
  - Frame 2: `The question requires specifying a single hand gesture from given frames of a short video. Unfortunately, the image provided does not include`
  - Frame 3: `SWIPE_LEFT
Description: The hand gesture captured in the video frame best fits the explanation "SWIPE`
  - Frame 4: `The hand gestures in question are characterized by the direction of movement of the wrist and fingers, which are central to identifying the`
  - Frame 5: `Swipe Left
<start of description>
The image is a still shot from a video and features a woman with`

### Video ID: 145116
- **True Label**: 2 (Rolling Hand Forward)
- **Final Mapped Label**: -1 (UNKNOWN)
- **Raw Outputs per Frame**:
  - Frame 1: `The hand gesture can be identified as follows: the index and middle fingers are extended and raised, which corresponds to the "`
  - Frame 2: `Based on the image and the task parameters, the label for the hand gesture is "STOP Sign". This is discerned`
  - Frame 3: `Based on the description of the hand gesture in the image, the most appropriate label would be "ROLLbate." The hand`
  - Frame 4: `The hand gesture made by the person in the image can be best described as a thumbs-down sign. This gesture is often`
  - Frame 5: `Rollellas
<start of description>
The image features a man standing against a plain background. He is middle-aged`

