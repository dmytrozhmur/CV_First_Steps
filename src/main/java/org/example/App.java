package org.example;

import nu.pattern.OpenCV;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;

/**
 * Hello world!
 *
 */
public class App {
    public static final String RAW_IMAGE_PATH = "src/main/resources/img.png";
    public static final String FACE_CLASSIFIER_PATH = "src/main/resources/face_classifier.xml";
    public static final String PROCESSED_IMAGE_PATH = "src/main/resources/image_processed.png";
    public static final double SCALE_FACTOR = 1.1;
    public static final int NEIGHBORS = 3;


    public static void main(String[] args ) {
        OpenCV.loadShared();

        Mat image = load();
        MatOfRect facesDetected = new MatOfRect();
        CascadeClassifier classifier = new CascadeClassifier();
        int minFaceSize = Math.round(image.rows() * 0.1f);
        classifier.load(FACE_CLASSIFIER_PATH);

        classifier.detectMultiScale(
                image,
                facesDetected,
                SCALE_FACTOR,
                NEIGHBORS,
                Objdetect.CASCADE_SCALE_IMAGE,
                new Size(minFaceSize, minFaceSize),
                new Size()
        );
        Rect[] facesArray = facesDetected.toArray();

        for (Rect face : facesArray) {
            Imgproc.rectangle(image, face.tl(), face.br(), new Scalar(0, 0, 255), 1);
        }
        save(image);
    }

    private static void save(Mat image) {
        Imgcodecs.imwrite(PROCESSED_IMAGE_PATH, image);
    }

    private static Mat load() {
        return Imgcodecs.imread(RAW_IMAGE_PATH);
    }
}
