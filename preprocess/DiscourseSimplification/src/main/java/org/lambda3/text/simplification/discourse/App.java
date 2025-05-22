/*
 * ==========================License-Start=============================
 * DiscourseSimplification : App
 *
 * Copyright © 2017 Lambda³
 *
 * GNU General Public License 3
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see http://www.gnu.org/licenses/.
 * ==========================License-End==============================
 */

package org.lambda3.text.simplification.discourse;

import org.lambda3.text.simplification.discourse.processing.DiscourseSimplifier;
import org.lambda3.text.simplification.discourse.processing.ProcessingType;
import org.lambda3.text.simplification.discourse.model.SimplificationContent;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.Collections; 
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class App {
    private static final org.slf4j.Logger LOGGER = LoggerFactory.getLogger(App.class);
    private static final DiscourseSimplifier DISCOURSE_SIMPLIFIER = new DiscourseSimplifier();

    private static void saveLines(File file, List<String> lines) {
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(file))) {
            bw.write(lines.stream().collect(Collectors.joining("\n")));

            // no need to close it.
            //bw.close()
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) throws IOException {
        // Read all lines from input.txt 
        List<String> inputLines = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader("input.txt"))) {
            String line;
            while ((line = br.readLine()) != null) {
                inputLines.add(line);
            }
        }

        // Process each line individually
        List<SimplificationContent> allContents = Collections.synchronizedList(new ArrayList<>());
        List<String> defaultFormatLines = Collections.synchronizedList(new ArrayList<>());
        List<String> flatFormatLines = Collections.synchronizedList(new ArrayList<>());

        // Then use parallelStream():
        inputLines.parallelStream().forEach(line -> {
            if (!line.trim().isEmpty()) {
                try {
                    File tempFile = File.createTempFile("temp-", ".txt");
                    saveLines(tempFile, Arrays.asList(line));
                    
                    SimplificationContent content = DISCOURSE_SIMPLIFIER.doDiscourseSimplification(
                        tempFile, ProcessingType.SEPARATE, true);
                    
                    // Synchronized adds (thread-safe)
                    allContents.add(content);
                    defaultFormatLines.add(content.defaultFormat(false));
                    flatFormatLines.add(content.flatFormat(false));
                    
                    tempFile.delete();
                } catch (IOException e) {
                    LOGGER.error("Error processing line: " + line, e);
                }
            }
        });

        // Write JSON output (all results combined)
        try (BufferedWriter bw = new BufferedWriter(new FileWriter("output.json"))) {
            bw.write("[");
            for (int i = 0; i < allContents.size(); i++) {
                // Use serializeToJSON with temporary file
                File tempJsonFile = File.createTempFile("temp-json-", ".json");
                allContents.get(i).serializeToJSON(tempJsonFile);
                
                // Read the temporary JSON file
                try (BufferedReader br = new BufferedReader(new FileReader(tempJsonFile))) {
                    String jsonLine;
                    while ((jsonLine = br.readLine()) != null) {
                        bw.write(jsonLine);
                    }
                }
                
                if (i < allContents.size() - 1) {
                    bw.write(",");
                }
                bw.newLine();
                
                // Delete temporary file
                tempJsonFile.delete();
            }
            bw.write("]");
        }

        // Use saveLines() for text outputs
        saveLines(new File("output_default.txt"), defaultFormatLines);
        saveLines(new File("output_flat.txt"), flatFormatLines);

        // SimplificationContent content = DISCOURSE_SIMPLIFIER.doDiscourseSimplification(new File("input.txt"), ProcessingType.SEPARATE, true);
        // content.serializeToJSON(new File("output.json"));
        // saveLines(new File("output_default.txt"), Arrays.asList(content.defaultFormat(false)));
        // saveLines(new File("output_flat.txt"), Arrays.asList(content.flatFormat(false)));
        LOGGER.info("done");
    }
}
