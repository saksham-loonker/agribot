import com.sakshyam.agribot.domain.model.BoundingBox;
import com.sakshyam.agribot.domain.scan.LeafObservation;
import com.sakshyam.agribot.domain.scan.PlantEvidence;
import com.sakshyam.agribot.domain.scan.PlantVerdict;
import com.sakshyam.agribot.domain.scan.VerdictKind;
import com.sakshyam.agribot.domain.scan.VerdictPolicy;
import com.sakshyam.agribot.ml.vision.EnsembleDecision;
import java.nio.file.*;
import java.util.*;

/** Execute actual compiled Kotlin decision classes on the saved model logits. */
public class VerifyKotlinReplay {
    public static void main(String[] args) throws Exception {
        double bias = Double.parseDouble(args[2]);
        double temperature = Double.parseDouble(args[3]);
        EnsembleDecision decision = new EnsembleDecision(9, 8, bias, temperature, null);
        Map<String,List<LeafObservation>> images = new LinkedHashMap<>();
        for (String line : Files.readAllLines(Path.of(args[0]))) {
            String[] fields = line.split("\t");
            List<LeafObservation> leaves = images.computeIfAbsent(fields[0], ignored -> new ArrayList<>());
            if (fields.length < 4) continue;
            List<float[]> members = new ArrayList<>();
            for (String member : fields[3].split("\\|")) {
                String[] numbers = member.split(",");
                float[] logits = new float[numbers.length];
                for (int i=0;i<numbers.length;i++) logits[i]=Float.parseFloat(numbers[i]);
                members.add(logits);
            }
            float[] probabilities = decision.combine(members).getProbabilities();
            leaves.add(new LeafObservation(new BoundingBox(0f,0f,1f,1f),Float.parseFloat(fields[2]),probabilities,Float.parseFloat(fields[1])));
        }
        List<String> output = new ArrayList<>();
        for (Map.Entry<String,List<LeafObservation>> image : images.entrySet()) {
            if (image.getValue().isEmpty()) {
                output.add(image.getKey()+"\tNO_LEAVES\tnull\t0\t0\tfalse"); continue;
            }
            PlantEvidence evidence = new PlantEvidence(9,1,8);
            String last = null; int stable = 0;
            for (int frame=1;frame<=14;frame++) {
                for (int i=0;i<image.getValue().size();i++) evidence.add(i+1L,frame,image.getValue().get(i));
                PlantVerdict verdict = evidence.verdict(new VerdictPolicy());
                String key = verdict.getKind().name()+":"+verdict.getLabelIndex();
                stable = key.equals(last) ? stable+1 : 0; last=key;
                if (verdict.getKind()!=VerdictKind.NEED_MORE_VIEWS &&
                    ((verdict.getKind()!=VerdictKind.UNSURE && stable>=2)||frame>=14)) {
                    output.add(image.getKey()+"\t"+verdict.getKind().name()+"\t"+verdict.getLabelIndex()+"\t"+frame+"\t"+verdict.getConfidence()+"\t"+verdict.getPartial());
                    break;
                }
            }
        }
        Files.write(Path.of(args[1]),output);
    }
}
