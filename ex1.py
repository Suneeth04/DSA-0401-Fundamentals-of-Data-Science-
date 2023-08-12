import numpy as np
student_scores = np.array([
    [85, 78, 92, 88],
    [92, 88, 76, 85],
    [78, 90, 83, 79],
    [62, 78, 90, 89]
])
average_scores = np.mean(student_scores, axis=0)
subject_with_highest_avg = np.argmax(average_scores)
print("Average scores for each subject:", average_scores)
print("Subject with the highest average score:", subject_with_highest_avg)
