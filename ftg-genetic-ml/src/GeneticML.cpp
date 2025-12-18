#include "GeneticML.h"

#include <algorithm>
#include <limits>


GeneticML::GeneticML(int populationSize, int generations)
    : currentGenomeIndex(1), currentGeneration(1), elitismPercent(0.2),
    maxGenerations(generations), populationSize(populationSize) {

    // init the random num gen
    std::random_device rd;
    random = std::mt19937(rd());

    // init the best with negative inf
    overallBestGenome.fitnessScore = -std::numeric_limits<float>::infinity();

    // create inital population (first generation)
    std::cout << "-- creating initial population" << "\n";
    for (int i = 0; i < populationSize; i++) {
        Genome g;
        g.id = i;
        g.fitnessScore = 0.0;

        // for params, starting with default with some mutations
        mutate(g.params);

        population.push_back(g); // add genome to the population
    }
}

Parameters& GeneticML::getNextGenomeParams() {
    if (currentGenomeIndex < static_cast<int>(population.size())) {
        return population[currentGenomeIndex].params; // gets the next genome (currentIndex-1 +1)
    }
    return population[0].params; // backup
}

void GeneticML::reportLapResult(float time, int crashCount) {
    if (currentGenomeIndex >= static_cast<int>(population.size())) return;

    // store lap data
    Lap l = {time, crashCount};
    population[currentGenomeIndex].laps.push_back(l);

    // calculate and store fitness score
    float score = calculateFitness(time, crashCount);
    population[currentGenomeIndex].fitnessScore = score;

    // log process

    // check and update bests
    if (score > overallBestGenome.fitnessScore) {
        overallBestGenome = population[currentGenomeIndex];
        std::cout << "New Record: " << time << "s, " << score << " pts \n";
    }

    // go to next index
    currentGenomeIndex++;
}

void GeneticML::evolveNextGeneration() {
    if (currentGeneration >= maxGenerations) {
        std::cout << "Max Generations (" << maxGenerations << ") Reached \n";
        return;
    }
    std::cout << "-- evolving generation to " << currentGeneration+1 << "\n";

    // archive current generation
    Iteration it;
    it.number = currentGeneration;
    it.genomes = population;
    history.push_back(it);

    // sort current population by fitness (best -> worst)
    std::sort(population.begin(), population.end(),
        [](const Genome& a, const Genome& b) {
            return a.fitnessScore > b.fitnessScore;
        });

    // start making the next generation
    std::vector<Genome> nextGen;

    // elitism (keep top performing params the same)
    int eliteCount = static_cast<int>(static_cast<float>(populationSize) * elitismPercent);
    for (int i = 0; i < eliteCount; i++) {
        Genome elite = population[i];
        elite.id = nextGen.size(); // renumber ID for new gen
        elite.laps.clear(); // clear old laps and fitness score
        elite.fitnessScore = 0.0f;
        nextGen.push_back(elite);
    }

    // fil lthe rest of the spots, by mixing (breeding)
    while(nextGen.size() < populationSize) {
        // pick two random parents from top 50%
        int limit = populationSize/2;
        int id1 = randomInt(0, limit);
        int id2 = randomInt(0, limit);

        // mix their params
        Parameters childParams = crossover(population[id1].params, population[id2].params);

        // mutate for more variation
        mutate(childParams);

        // create child
        Genome child;
        child.id = nextGen.size();
        child.params = childParams;
        child.fitnessScore = 0.0f;
        nextGen.push_back(child);
    }

    // swap from old to new population
    population = nextGen;
    currentGeneration++;
    currentGenomeIndex = 0;
}

bool GeneticML::isGenerationComplete() const {
    return currentGenomeIndex >= population.size();
}


float GeneticML::calculateFitness(float time, int crashCount) {
    // faster better, huge deduction for crashing

    if (time <= 0.1) return 0.0;

    float speedScore = 1000.0f / time;
    float penalty = static_cast<float>(crashCount) * 500.0f;

    return speedScore - penalty;
}

Parameters GeneticML::crossover(const Parameters& p1, const Parameters& p2) {
    Parameters child = p1;

    // crossover 50% chance for each gene
    if(randomFloat(0,1) > 0.5f) child.bubbleRadius = p2.bubbleRadius;
    if(randomFloat(0,1) > 0.5f) child.preprocessConvSize = p2.preprocessConvSize;
    if(randomFloat(0,1) > 0.5f) child.bestPointConvSize = p2.bestPointConvSize;
    if(randomFloat(0,1) > 0.5f) child.maxLidarDist = p2.maxLidarDist;
    if(randomFloat(0,1) > 0.5f) child.straightSpeed = p2.straightSpeed;
    if(randomFloat(0,1) > 0.5f) child.cornerSpeed = p2.cornerSpeed;
    if(randomFloat(0,1) > 0.5f) child.centerBiasAlpha = p2.centerBiasAlpha;
    if(randomFloat(0,1) > 0.5f) child.edgeGuardDeg = p2.edgeGuardDeg;
    if(randomFloat(0,1) > 0.5f) child.steerSmoothAlpha = p2.steerSmoothAlpha;

    return child;
}

void GeneticML::mutate(Parameters &p) {
}

float GeneticML::randomFloat(float min, float max) {
    std::uniform_real_distribution<float> dist(min, max);
    return dist(random);
}

int GeneticML::randomInt(int min, int max) {
    std::uniform_int_distribution<int> dist(min, max);
    return dist(random);
}

Genome GeneticML::getBestGenome() const { return overallBestGenome; }

int GeneticML::getCurrentGeneration() const { return currentGeneration; }

int GeneticML::getCurrentGenomeIndex() const { return currentGenomeIndex; }
