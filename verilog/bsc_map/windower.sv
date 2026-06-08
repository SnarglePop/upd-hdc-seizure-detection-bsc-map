`timescale 1ns / 1ps
module windower (
    clk,
    nrst,
    en,
    samples,
    done,
    sample_memory
);

    parameter NUM_CHS = 4;
    parameter WINDOW_SIZE = 256;
    parameter WINDOW_STEP = 128;
    parameter SAMPLE_SIZE = 16;
    

    input clk;
    input nrst;
    input en;
    input [NUM_CHS*SAMPLE_SIZE - 1: 0] samples;
    output reg done;
    output reg [WINDOW_SIZE*NUM_CHS*SAMPLE_SIZE - 1: 0] sample_memory;
    
    reg [WINDOW_SIZE*NUM_CHS*SAMPLE_SIZE - 1: 0] running_memory;


    reg isFirstWindow;
    reg isFirstSample;
    reg isLBP;
    reg [$clog2(WINDOW_SIZE): 0] sampleCounter;


    always_ff @(posedge clk or negedge nrst) begin
        if (!nrst) begin
            done <= 0;
            sample_memory <= 0;
            running_memory <= 0;

            isFirstWindow <= 1;
            isFirstSample <= 1;
            isLBP <= 1;
            sampleCounter <= 0;
        end else begin
            if (en) begin
                // Put samples into sample memory
                running_memory[(WINDOW_SIZE - 1)*SAMPLE_SIZE*NUM_CHS +: SAMPLE_SIZE*NUM_CHS] <= samples;

                // Shift sample memory
                for (int i = 0; i < WINDOW_SIZE - 1; i = i + 1)
                    running_memory[i*SAMPLE_SIZE*NUM_CHS +: SAMPLE_SIZE*NUM_CHS] <= running_memory[(i + 1)*SAMPLE_SIZE*NUM_CHS +: SAMPLE_SIZE*NUM_CHS];
                
                // Window generation
                if (sampleCounter == WINDOW_STEP - 1) begin
                    sampleCounter <= 0;
                    if (isFirstWindow) begin
                        isFirstWindow <= 0;
                    end else begin
                        sample_memory <= running_memory;
                        done <= 1;
                    end
                end
                else if (isFirstSample) begin
                    isFirstSample <= 0;
                end
                else if (isLBP) begin
                    if (sampleCounter == 5) begin
                        isLBP <= 0;
                        sampleCounter <= 0;
                    end
                    else begin
                        sampleCounter <= sampleCounter + 1;
                    end
                end
                else begin
                    sampleCounter <= sampleCounter + 1;
                    done <= 0;
                end
            end
            else
                done <= 0;
        end
    end

endmodule