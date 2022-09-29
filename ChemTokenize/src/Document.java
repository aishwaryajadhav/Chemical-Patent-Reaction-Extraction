import uk.ac.cam.ch.wwmm.oscar.document.Token;
import uk.ac.cam.ch.wwmm.oscar.document.TokenSequence;

import java.util.ArrayList;
import java.util.List;

public class Document {
    int count;
    List<List<String>> paragraphs;
    Document(){
        count = 0;
        paragraphs = new ArrayList<>();
    }
    void addPara(List<String> para_tokens){
        paragraphs.add(para_tokens);
        count++;
    }
}
