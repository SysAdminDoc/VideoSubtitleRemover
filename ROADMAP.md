# Video Subtitle Remover Pro -- Roadmap

Remaining-work backlog only: every item below is NOT yet implemented.
Completed items are deleted from this file; history lives in CHANGELOG.md and git.



### Simplification

117. **OpenCV 5 DNN for detection models** -- evaluate running RapidOCR's
     PP-OCR ONNX models through `cv2.dnn` instead of ONNX Runtime; if
     viable, the core detect+inpaint pipeline needs only OpenCV 5 + numpy.
     Priority: P2 simplification. Effort: L. Confidence: medium.
     Acceptance criteria:
     - PP-OCR detection and recognition models load and run via cv2.dnn
       with accuracy parity on reference clips.
     - ONNX Runtime path kept as fallback.
     Source: https://opencv.org/opencv-5/

---



## Research-Driven Additions

### P1 -- Trust and release readiness

### P2 -- Dependency, documentation, and UX hardening

### P3 -- Research bench
