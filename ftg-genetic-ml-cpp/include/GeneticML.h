#ifndef GENETICML_H
#define GENETICML_H

#include <iostream>
#include <vector>
#include "Parameters.h"
#include <random>

// class manages both the improvement and changing of parameters for Genetic ML model
// and also handles getting and storing the lap times.

// Lap structure to store individual lap information
struct Lap {
    float time; // lap time
    int crashCount; // number of times hit the wall
};

// Represent a single candidate driver config
struct Genome {
    int id;                // unique id
    Parameters params;     // different params
    float fitnessScore;    // calculated performance score
    std::vector<Lap> laps; // all laps driven by this genome
};

// represents a full generation of evolution
struct Iteration {
    int number;
    std::vector<Genome> genomes;
};


class GeneticML {
public:
    // Constructor
    GeneticML(int populationSize, int generations);

    // Core Functions
    Parameters& getNextGenomeParams();                // get the params (ref) for the next genome
    void reportLapResult(float time, int crashCount); // called after a lap finished to report how it went, process results

    // Genetic Algorithm Steps
    void evolveNextGeneration();              // called when the current generation is complete
    bool isGenerationComplete() const;        // checks if every genome in current generation is run

    // utils
    Genome getBestGenome() const;
    int getCurrentGeneration() const;
    int getCurrentGenomeIndex() const;

private:
    // Internal Functions
    float calculateFitness(float time, int crashCount); // calcs the fitness function that defines good
    Parameters crossover(const Parameters &p1, const Parameters &p2); // mix the two parent parameters
    void mutate(Parameters& p);

    // Random utils
    float randomFloat(float min, float max);
    int randomInt(int min, int max);

    // State Variables
    std::vector<Genome> population;   // current generation
    std::vector<Iteration> history;   // history of past generations

    int currentGenomeIndex;
    int currentGeneration;
    int maxGenerations;
    int populationSize;
    float elitismPercent; // decimal percent of best performing params to keep unchanged

    Genome overallBestGenome;

    std::mt19937 random; // random number generator

};



#endif //GENETICML_H
