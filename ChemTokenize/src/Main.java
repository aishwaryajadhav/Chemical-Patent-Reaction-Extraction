
import com.google.gson.Gson;
import uk.ac.cam.ch.wwmm.oscar.document.Token;
import uk.ac.cam.ch.wwmm.oscar.document.TokenSequence;
import uk.ac.cam.ch.wwmm.oscartokeniser.Tokeniser;
import org.apache.commons.io.FilenameUtils;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        System.out.println("Hello world!");

        Tokeniser tokeniser = Tokeniser.getDefaultInstance();

        File dev_path = new File("C:\\Users\\meais\\Documents\\CMU\\Independent Study\\ReactionExtraction\\data\\dev");
        File train_path = new File("C:\\Users\\meais\\Documents\\CMU\\Independent Study\\ReactionExtraction\\data\\train");


        String output_dev_path = "C:\\Users\\meais\\Documents\\CMU\\Independent Study\\ReactionExtraction\\data\\json\\dev";
        String output_train_path = "C:\\Users\\meais\\Documents\\CMU\\Independent Study\\ReactionExtraction\\data\\json\\train";

        for (final File fileEntry : dev_path.listFiles()) {
            if (!fileEntry.isDirectory() && fileEntry.getName().endsWith(".txt")) {
                System.out.println(fileEntry.getName());
                processFile(fileEntry, output_dev_path, tokeniser);

            }
        }
    }
    public static void processFile(File file, String output_dir, Tokeniser tokeniser){
        Document document = new Document();
        try {
            Scanner scanner = new Scanner(file);
            while (scanner.hasNextLine()) {
                TokenSequence tokenized_string = tokeniser.tokenise(scanner.nextLine());
                document.addPara(tokenized_string.getTokenStringList());
            }
            scanner.close();
            Gson gson = new Gson();
            String output_file = output_dir + "\\" +FilenameUtils.removeExtension(file.getName())+".json";
            System.out.println(output_file);

            Writer writer = new FileWriter(output_file);
            gson.toJson(document, writer);
            writer.flush(); //flush data to file   <---
            writer.close();

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}