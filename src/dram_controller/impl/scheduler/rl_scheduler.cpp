#include <vector>
#include <map>
#include <random>
#include <algorithm>
#include <iostream>

#include "base/base.h"
#include "dram_controller/controller.h"
#include "dram_controller/scheduler.h"

namespace Ramulator {

struct State {
    int numReadRequests;
    int numWriteRequests;
    //bool rowOpen;
    bool operator<(const State& other) const {
        if (numReadRequests != other.numReadRequests) {
            return numReadRequests < other.numReadRequests;
        }
        return numWriteRequests < other.numWriteRequests;
    }
};

enum Action {READ, WRITE};

class RLScheduler : public IScheduler, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IScheduler, RLScheduler, "RLScheduler", "RL-based DRAM Scheduler.")
  private:
    IDRAM* m_dram;
    std::map<State, std::map<Action, double>> qTable;  // Q-table
    double learningRate = 0.1;
    double discountFactor = 0.95;
    double epsilon = 0.7;  // For ε-greedy policy
    State currentState;
    Action nextAction;

  
  public:
    void init() override { };

    void setup(IFrontEnd* frontend, IMemorySystem* memory_system) override {
      m_dram = cast_parent<IDRAMController>()->m_dram;
      qTable[State{0, 0}] = {{READ, 0.0}, {WRITE, 0.0}};
      qTable[State{0, 1}] = {{READ, 0.0}, {WRITE, 0.0}};
      qTable[State{1, 0}] = {{READ, 0.0}, {WRITE, 0.0}};
      qTable[State{1, 1}] = {{READ, 0.0}, {WRITE, 0.0}};
    };

    ReqBuffer::iterator compare(ReqBuffer::iterator req1, ReqBuffer::iterator req2) override {
        bool ready1 = m_dram->check_ready(req1->command, req1->addr_vec);
        bool ready2 = m_dram->check_ready(req2->command, req2->addr_vec);
        if (req1->type_id == Request::Type::Read && req2->type_id == Request::Type::Write) {
            if (nextAction == READ) {
                return req1;
            } else {
                return req2;
            }
        } else if (req1->type_id == Request::Type::Write && req2->type_id == Request::Type::Read) {
            if (nextAction == WRITE) {
                return req1;
            } else {
                return req2;
            }
        } 
    }
    
    
   ReqBuffer::iterator get_best_request(ReqBuffer& buffer) override {
        if (buffer.size() == 0) {
            return buffer.end();
        }
        std::cout << "RLScheduler: get_best_request" << std::endl;

        // Example logic to select the best request based on the RL policy
        currentState = observeState(buffer);
        std::cout << "Current state: " << currentState.numReadRequests << " " << currentState.numWriteRequests << std::endl;
        nextAction = selectAction(currentState);
        std::cout << "Selected action: " << nextAction << std::endl;
        double reward = executeAction(nextAction);
        std::cout << "Reward: " << reward << std::endl;
        State newState = observeState(buffer);
        std::cout << "New state: " << newState.numReadRequests << " " << newState.numWriteRequests << std::endl;
        updateQTable(currentState, nextAction, reward, newState);
        std::cout << "Q-table updated" << std::endl;
        currentState = newState;
        std::cout << "New current state: " << currentState.numReadRequests << " " << currentState.numWriteRequests << std::endl;


        for (auto& req : buffer) {
            req.command = m_dram->get_preq_command(req.final_command, req.addr_vec);
        }
        // Select the best request based on the selected action
        auto candidate = buffer.begin();
        for (auto next = std::next(buffer.begin()); next != buffer.end(); next++) {
            candidate = compare(candidate, next);
        }
        std::cout << "Best request: " << candidate->type_id  << candidate->addr << std::endl;
        return candidate;
    }


    State observeState(ReqBuffer& buffer) {
        State state;
        state.numReadRequests = std::count_if(buffer.begin(), buffer.end(), 
                                            [](const Request& req) { return req.type_id == Request::Type::Read; });
        state.numWriteRequests = std::count_if(buffer.begin(), buffer.end(), 
                                            [](const Request& req) { return req.type_id == Request::Type::Write; });
        return state;
    }

    Action selectAction(const State& state) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);

        if (dis(gen) < epsilon) {
            std::uniform_int_distribution<> action_dis(0, 1); 
            return static_cast<Action>(action_dis(gen));
        } else {
            auto it = std::max_element(qTable[state].begin(), qTable[state].end(),
                                    [](const auto& a, const auto& b) { return a.second < b.second; });
            return it->first;
        }
    }

    double executeAction(Action action) {
        double reward = 0.0;
        switch (action) {
            case READ:
                reward = 0.5; 
                break;
            case WRITE:
                reward = 0.5;
                break;
            // case ACTIVATE:
            //     reward = 0.8; 
            //     break;
            // case PRECHARGE:
            //     reward = 0.2; 
            //     break;
        }
        return reward;
    }

    void updateQTable(const State& oldState, Action action, double reward, const State& newState) {
        Action nextAction = selectAction(newState); 
        double oldQValue = qTable[oldState][action];
        double newQValue = oldQValue + learningRate * (reward + discountFactor * qTable[newState][nextAction] - oldQValue);
        qTable[oldState][action] = newQValue;
    }

};

}       // namespace Ramulator
