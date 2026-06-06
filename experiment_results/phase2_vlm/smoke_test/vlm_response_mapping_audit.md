# VLM Response Mapping Audit Report

Analyzed all **82** smoke test samples where the legacy parser returned `predicted_label = -1` (UNKNOWN).

## Why the Legacy Parser Failed
The legacy parser only checked the first line of the VLM output and looked for exact matches of label tokens (`SWIPE_LEFT`, etc.). However, FastVLM-1.5B frequently:
1. Generates conversational prefaces (e.g., *'The label that best describes... is STOP SIGN'*).
2. Explains that the image lacks visibility or contains no hands.
3. Hallucinates suffixes on label words (e.g., *'ROLLPlayable'*, *'ROLLversed'*, *'STOP SIGNGesture'*).

By only matching exact tokens on the first line, frames with conversational labels were mapped to `-1`, dragging down the majority-vote aggregator to `UNKNOWN`.

## Detailed Audit Log (First 15 Failures)
| Video ID | True Label | Raw VLM Response (Frame-by-Frame) | Legacy Mapped | Improved Mapped | Parser Failure Reason |
|---|---|---|---|---|---|
| 122832 | ROLL_FWD | F1: Based on the provided image, there is no visible hand gesture to analyze, as it ...<br>F2: SWIPE_LEFT
In response to the task, I must identify the hand gesture while ignor...<br>F3: The image provided does not show any hands performing any defined hand gestures....<br>F4: The label that best describes the hand gesture in the image is "STOP signing", a...<br>F5: The label that describes the hand gesture based on the given options, with the h... | UNKNOWN | UNKNOWN | Conversational output or suffix hallucination |
| 129972 | SWIPE_RIGHT | F1: I'm sorry, but you have not provided a specific video to analyze. If you have a ...<br>F2: The hand gesture is best described by the label "STOP Signing." This label refer...<br>F3: ROLLumont
The label that best describes the hand gesture in the current frames i...<br>F4: The label that best describes the hand gesture in the list provided is "ROLLPlay...<br>F5: SWIPE_LEFT
This label is the most suitable for a single hand gesture in an image... | UNKNOWN | ROLL_FWD | Model uncertainty/refusal (legitimate UNKNOWN) |
| 100526 | STOP_SIGN | F1: The image provided does not align with the context of the question, as it contai...<br>F2: The label that would best describe the hand gesture in the image is ROLLTyped. T...<br>F3: Rollversed
<start of description>
The image features the upper body of a person ...<br>F4: The hand gesture in the image matches the 'STOP Sign' hand signal, which is a un...<br>F5: The appropriate label for the image is "STOP Signing." This is deduced from the ... | UNKNOWN | ROLL_FWD | Conversational output or suffix hallucination |
| 80870 | ROLL_FWD | F1: The question seems to refer to a process known as frame analysis, typically used...<br>F2: The question requires specifying a single hand gesture from given frames of a sh...<br>F3: SWIPE_LEFT
Description: The hand gesture captured in the video frame best fits t...<br>F4: The hand gestures in question are characterized by the direction of movement of ...<br>F5: Swipe Left
<start of description>
The image is a still shot from a video and fea... | UNKNOWN | UNKNOWN | Conversational output or suffix hallucination |
| 145116 | ROLL_FWD | F1: The hand gesture can be identified as follows: the index and middle fingers are ...<br>F2: Based on the image and the task parameters, the label for the hand gesture is "S...<br>F3: Based on the description of the hand gesture in the image, the most appropriate ...<br>F4: The hand gesture made by the person in the image can be best described as a thum...<br>F5: Rollellas
<start of description>
The image features a man standing against a pla... | UNKNOWN | UNKNOWN | Conversational output or suffix hallucination |
| 137514 | SWIPE_LEFT | F1: The label that best describes the hand gesture in the given image is "STOP(sign)...<br>F2: SWIPE_LEFT

The label best fitting the hand gesture in this image is "SWIPE_LEFT...<br>F3: Based on the provided image, the label text that best describes the hand gesture...<br>F4: The label that best describes the hand gesture in the image is "SWIPE_LEFT." Thi...<br>F5: The task requires identifying a single hand gesture. Unfortunately, due to the i... | UNKNOWN | STOP_SIGN | Conversational output or suffix hallucination |
| 122698 | SWIPE_RIGHT | F1: Given the guidelines and the task to identify the hand gesture without focusing ...<br>F2: The label that best describes the hand gesture, focusing solely on the hands as ...<br>F3: The task is to identify one hand gesture out of the provided list, based on the ...<br>F4: The appropriate label for the hand gesture in this image is "STOP Signs." This l...<br>F5: The provided image is a close-up of a person's face, and there are no hand gestu... | UNKNOWN | UNKNOWN | Model uncertainty/refusal (legitimate UNKNOWN) |
| 19927 | SWIPE_LEFT | F1: The question pertains to a "hand gesture" from a visual medium, specifically foc...<br>F2: The label that best describes the hand gesture in the one photograph you've sent...<br>F3: The label that best describes the hand gesture in the image is "STOP SIGN". The ...<br>F4: Based on the description of the hand gesture in the video frames, a single hand ...<br>F5: The image provided is a close-up of a single individual's face with the focus on... | UNKNOWN | UNKNOWN | Conversational output or suffix hallucination |
| 39995 | SWIPE_LEFT | F1: I'm sorry, but without visual content, I cannot provide a description or categor...<br>F2: Based on your request to focus solely on the hand gestures without considering t...<br>F3: Based on the image provided, the hand gesture cannot be determined as a single h...<br>F4: Based on the description of the hand gesture in the frames, which shows the indi...<br>F5: The label that best fits the hand gesture depicted in the image is "STOP Sign." ... | UNKNOWN | UNKNOWN | Model uncertainty/refusal (legitimate UNKNOWN) |
| 141571 | ROLL_FWD | F1: The hand gesture in question best fits the description of "STOP signer." This ge...<br>F2: I'm sorry, but I cannot comply with your request as it involves identifying or m...<br>F3: The label that best describes the hand gesture is "SWIPE_RIGHT," as the right ha...<br>F4: The hand gesture in the image cannot be classified into any of the specified opt...<br>F5: Unknown
Please note that the image description and the provided video frames are... | UNKNOWN | UNKNOWN | Model uncertainty/refusal (legitimate UNKNOWN) |
| 133144 | ROLL_FWD | F1: ROLLointed
<start of description>
The image is a close-up of a person who seems ...<br>F2: STOP Sign

 The post was intended to be an AI-generated image based on provided ...<br>F3: After the analysis of the hand gesture in the provided image, the best label fro...<br>F4: The requested label cannot be determined from the current view of the image prov...<br>F5: ROLLosciPEACE
This hand gesture is one hand held slightly above the other, with ... | UNKNOWN | ROLL_FWD | Conversational output or suffix hallucination |
| 74444 | STOP_SIGN | F1: I'm sorry, but you've provided an image with a person and a video frame, but I c...<br>F2: The image falls under the category of the "UNKNOWN" label, as it does not depict...<br>F3: STOP Selection
The hand gesture in question seems to depict a 'stop' action due ...<br>F4: The label that best describes the hand gesture in the image is "STOP_sign." The ...<br>F5: The hand gesture in the image is most accurately described by the label "SWIPE_L... | UNKNOWN | UNKNOWN | Model uncertainty/refusal (legitimate UNKNOWN) |
| 35103 | ROLL_FWD | F1: The label that best describes the hand gesture in this image is "SWIPE_LEFT," as...<br>F2: Since the image does not include the hands or any gestures that could be classif...<br>F3: Based on the criteria provided, the hand gesture of the individual cannot be det...<br>F4: SWIPE_RIGHT
Explanation: The hand gesture in the image shows the right hand movi...<br>F5: ROLL.advanced
The label that best describes the hand gesture in this context is ... | UNKNOWN | UNKNOWN | Conversational output or suffix hallucination |
| 38093 | SWIPE_LEFT | F1: Given the task description, we must select one label from the list that best des...<br>F2: The task requires selecting a hand gesture label from the provided options, focu...<br>F3: The task requires identifying the hand gesture of a single individual in a frame...<br>F4: The hand gesture that the person is performing, visible even though the face isn...<br>F5: The task requires identifying a hand gesture from a short video frame based on i... | UNKNOWN | UNKNOWN | Conversational output or suffix hallucination |
| 81277 | SWIPE_RIGHT | F1: The image is too small and lacks visibility, making it impossible to determine t...<br>F2: ROLLrollers
**
**
**
**
**
**
**...<br>F3: The given task requires the identification of a hand gesture from a sequence of ...<br>F4: Swipe Left
This determination is made by observing the orientation and apparent ...<br>F5: ROLL advancement
<start of description>
The image is a close-up of a person with... | UNKNOWN | UNKNOWN | Conversational output or suffix hallucination |
