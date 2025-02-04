```python
# This is the main file in a markdown format, with example output. The images are provided in this directory.

import os

# Data manipulation and analysis.
import pandas as pd

# Has multi-dimensional arrays and matrices.
import numpy as np

# Data visualization tools.
import seaborn as sns

import random
import secrets

import matplotlib.pyplot as plt

# Used for efficiency checks
import timeit

# Data visualization
import networkx as nx

# Using mesa module for making the ABM
import mesa

# Data visualization interactive
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scipy.sparse import csr_matrix

# Required dep for visualization
import nbformat

# Multithreading go brr
import threading
```


```python
# # Global variables
# influence_rate = 0.1  # η - Controls the speed of opinion change
# external_influence_strength = 0.3  # α - Strength of external propaganda
# opinion_difference_sensitivity = 0.6  # β - Sensitivity to opinion differences for link formation
# break_threshold = 0.3  # Δ_break - Threshold for breaking links
# propaganda_initial_sensitivity = 0.9  # Sensitivity to propaganda agents used for creating initial links
# propaganda_sensitivity = 1 # Sensitivity to propaganda agents used for breaking links with hubs
# current_step=0
# revolt_threshold = 0.8  # Threshold for revolt
# neighbor_divergence_threshold = 0.7  # Threshold for breaking links with neighbors after revolt
# neighbor_similarity_threshold = 0.3  # Threshold for forming links with new individuals after revolt

# Constants
INFLUENCE_RATE = 0.1
EXTERNAL_INFLUENCE_STRENGTH = 0.3
OPINION_DIFF_SENSITIVITY = 0.6
BREAK_THRESHOLD = 0.3
PROPAGANDA_INITIAL_SENSITIVITY = 0.9
PROPAGANDA_SENSITIVITY = 1.0
REVOLT_THRESHOLD = 0.8
NEIGHBOR_DIVERGENCE_THRESHOLD = 0.7
NEIGHBOR_SIMILARITY_THRESHOLD = 0.3
CURRENT_STEP = 0
```


```python
# Create empty NetworkX graph
graph = nx.Graph()
```


```python
# Class for individual, based on agent.
# Contains the required information that forms the basis of interactions
# The interactions are their own classes that happen at each time-step

class IndividualAgent(mesa.Agent):
    def __init__(self, unique_id, model):
        super().__init__(model)
        self.unique_id = unique_id
        self.name = f"Individual {unique_id}"
        self.opinion = np.random.uniform(-1, 1)
        self.resistance = np.random.uniform(0, 1)
        self.influence = np.random.uniform(0.8, 1.5)
        self.revolt_threshold = np.random.uniform(0.7, 1)

    def say_hi(self):
        print(f"Hi, I am {self.name} with opinion {self.opinion}, resistance {self.resistance}, influence {self.influence}, and revolt threshold {self.revolt_threshold}")
```


```python
# Class for individual, based on agent.
# Contains the required information that forms the basis of interactions
# The interactions are their own classes that happen at each time-step

class PropagandaHub(mesa.Agent):
    def __init__(self, unique_id, model, opinion, influence_radius, influence_strength):
        super().__init__(model)
        self.unique_id = unique_id
        self.name = f"Propaganda Hub {unique_id}"
        self.opinion = opinion
        self.influence_radius = influence_radius
        self.influence_strength = influence_strength

    def say_hi(self):
        print(f"Hi, I am an propaganda agent, you can call me {self.name} and I have the id of {str(self.unique_id)}. \
      I have the opinion of {self.opinion}, and influence strength is {self.influence_strength}\
      with a radius of {self.influence_radius}.")

    
```


```python
class ExperimentModel(mesa.Model):
    def __init__(self, n_individuals, n_propaganda_hubs):
        super().__init__()
        self.num_individuals = n_individuals
        self.num_propaganda_hubs = n_propaganda_hubs
        self.schedule = mesa.time.RandomActivation(self)
        self.graph = nx.Graph()
        self.metrics = {
            "opinion_distribution": [],
            "polarization_index": [],
            "cascade_thresholds": [],
            "average_degree": [],
            "revolt_frequency": []
        }

        # Create agents
        self.individuals = [IndividualAgent(i, self) for i in range(n_individuals)]
        self.hubs = [
            PropagandaHub(i + n_individuals, self, np.random.choice([-1, 1]),
                          np.random.uniform(10, 100), np.random.uniform(0.4, 1))
            for i in range(n_propaganda_hubs)
        ]

        # Add agents to scheduler and graph
        for agent in self.individuals + self.hubs:
            self.schedule.add(agent)
            self.graph.add_node(agent.unique_id)

        # Create edges
        self.create_edges()

        # Data structures for optimization
        self.individual_opinions = np.array([agent.opinion for agent in self.individuals])
        self.resistances = np.array([agent.resistance for agent in self.individuals])
        self.influences = np.array([agent.influence for agent in self.individuals])
        self.hub_opinions = np.array([hub.opinion for hub in self.hubs])
        self.hub_strengths = np.array([hub.influence_strength for hub in self.hubs])
        self.hub_radii = np.array([hub.influence_radius for hub in self.hubs])
        self.adjacency_matrix = nx.to_scipy_sparse_array(self.graph, format="csr")

    def create_edges(self):
        # Connect individuals probabilistically
        individual_ids = [agent.unique_id for agent in self.individuals]
        opinions = np.array([agent.opinion for agent in self.individuals])
        differences = np.abs(opinions[:, None] - opinions)
        connection_probs = (np.random.rand(len(opinions), len(opinions)) < 0.1) & (differences < OPINION_DIFF_SENSITIVITY)
        np.fill_diagonal(connection_probs, 0)
        edges = np.argwhere(connection_probs)
        self.graph.add_edges_from([(individual_ids[i], individual_ids[j]) for i, j in edges])

        # Connect hubs to individuals based on opinion similarity
        hub_ids = [hub.unique_id for hub in self.hubs]
        hub_opinions = np.array([hub.opinion for hub in self.hubs])
        hub_individual_differences = np.abs(hub_opinions[:, None] - opinions)
        hub_connection_probs = (hub_individual_differences < PROPAGANDA_INITIAL_SENSITIVITY)
        for i, hub_id in enumerate(hub_ids):
            connections = np.argwhere(hub_connection_probs[i]).flatten()
            self.graph.add_edges_from([(hub_id, individual_ids[j]) for j in connections])

        # Update adjacency matrix
        self.adjacency_matrix = nx.to_scipy_sparse_array(self.graph, format="csr")


    def track_metrics(self):
        """Track metrics at each step."""
        # Opinion distribution (track opinions only for individuals)
        opinions = self.individual_opinions.copy()
        self.metrics["opinion_distribution"].append(opinions)

        # Polarization index
        polarization_index = np.mean(np.abs(opinions))
        self.metrics["polarization_index"].append(polarization_index)

        # Cascade thresholds
        if len(self.metrics["opinion_distribution"]) > 1:
            prev_opinions = self.metrics["opinion_distribution"][-2]
            cascades = np.sum(np.abs(opinions - prev_opinions) > 0.1)  # Example threshold
            self.metrics["cascade_thresholds"].append(cascades)
        else:
            self.metrics["cascade_thresholds"].append(0)

        # Network dynamics (average degree)
        degrees = [degree for _, degree in self.graph.degree()]
        avg_degree = np.mean(degrees)
        self.metrics["average_degree"].append(avg_degree)

        # Revolt frequency
        revolt_count = np.sum(opinions > REVOLT_THRESHOLD)  # Count revolts (as an example)
        self.metrics["revolt_frequency"].append(revolt_count)

    def step(self):
        """Advance the model by one step."""

        global CURRENT_STEP
        CURRENT_STEP += 1
        print(f"Current step is {CURRENT_STEP}!")


        adjacency = self.adjacency_matrix
    
        # Pad opinions and influences to include zeros for hubs
        padded_opinions = np.concatenate([self.individual_opinions, np.zeros(self.num_propaganda_hubs)])
        padded_influences = np.concatenate([self.influences, np.zeros(self.num_propaganda_hubs)])
    
        # --- Influence Calculations ---
        neighbor_influences = adjacency.dot(padded_opinions * padded_influences)
        neighbor_weights = adjacency.dot(padded_influences)
        neighbor_effect = np.divide(neighbor_influences, neighbor_weights, where=neighbor_weights != 0) - padded_opinions
    
        # Propaganda influence
        # Pad hub_opinions with zeros to match the number of individuals
        padded_hub_opinions = np.concatenate([np.zeros(self.num_individuals), self.hub_opinions])
    
        # Calculate propaganda effect using the padded array
        propaganda_effect = adjacency.dot(padded_hub_opinions * EXTERNAL_INFLUENCE_STRENGTH)
    
        # Total opinion updates (only for individuals)
        total_effect = INFLUENCE_RATE * (neighbor_effect[:self.num_individuals] + propaganda_effect[:self.num_individuals])
        self.individual_opinions = np.clip(self.individual_opinions + total_effect, -1, 1)
    
        # --- Network Updates ---
        opinion_diffs = np.abs(self.individual_opinions[:, None] - self.individual_opinions)
        break_links = (opinion_diffs > BREAK_THRESHOLD).astype(int)
        
        # Pad break_links with zeros to match the adjacency matrix dimensions
        padded_break_links = np.zeros(self.adjacency_matrix.shape)
        break_links_sparse = csr_matrix(padded_break_links)
        padded_break_links[:self.num_individuals, :self.num_individuals] = break_links

    
        # Update adjacency matrix using the padded break_links
        self.adjacency_matrix = adjacency - break_links_sparse
    
        # --- Revolt Mechanics ---
        divergences = np.abs(
            adjacency.dot(padded_opinions)[:self.num_individuals] /
            adjacency.sum(axis=1).flatten()[:self.num_individuals] - 
            self.individual_opinions
        )
        revolts = divergences > REVOLT_THRESHOLD
        for i, revolted in enumerate(revolts):
            if revolted:
                # Break all connections for the revolted agent
                self.adjacency_matrix[i] = 0
    
                # Shift opinion to an extreme based on dominant influence
                dominant_influence = 1 if np.sum(self.adjacency_matrix[i]) > 0 else -1
                self.individual_opinions[i] = dominant_influence
    
                # Create new ties with agents holding similar opinions
                similar_agents = np.where(np.abs(self.individual_opinions - self.individual_opinions[i]) < NEIGHBOR_SIMILARITY_THRESHOLD)[0]
                for j in similar_agents:
                    self.adjacency_matrix[i, j] = 1
                    self.adjacency_matrix[j, i] = 1
    
        # Update the graph
        self.graph = nx.from_scipy_sparse_array(self.adjacency_matrix, create_using=nx.Graph)

        self.track_metrics()


        # Add metadata for nodes
        for agent in self.schedule.agents:
            if agent.unique_id in self.graph:
                # Add metadata depending on the agent type
                if isinstance(agent, IndividualAgent):
                    self.graph.nodes[agent.unique_id].update({
                        "type": "Individual",
                        "opinion": agent.opinion,
                        "resistance": agent.resistance,
                        "influence": agent.influence,
                        "revolt_threshold": agent.revolt_threshold,
                    })
                elif isinstance(agent, PropagandaHub):
                    self.graph.nodes[agent.unique_id].update({
                        "type": "PropagandaHub",
                        "opinion": agent.opinion,
                        "influence_radius": agent.influence_radius,
                        "influence_strength": agent.influence_strength,
                    })

        # Save the graph in GraphML format
        output_dir = "graphml_exports"
        os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
        filename = os.path.join(output_dir, f"graph_step_{self.schedule.steps}.graphml")
        nx.write_graphml(self.graph, filename)

        self.schedule.step()


    def visualize_metrics(self):
        """Visualize all tracked metrics."""
        steps = len(self.metrics["polarization_index"])

        # Polarization Index
        plt.figure()
        plt.plot(range(steps), self.metrics["polarization_index"])
        plt.title("Polarization Index Over Time")
        plt.xlabel("Step")
        plt.ylabel("Polarization Index")
        plt.show()

        # Cascade Thresholds
        plt.figure()
        plt.plot(range(steps), self.metrics["cascade_thresholds"])
        plt.title("Cascade Thresholds Over Time")
        plt.xlabel("Step")
        plt.ylabel("Cascade Count")
        plt.show()

        # Average Degree
        plt.figure()
        plt.plot(range(steps), self.metrics["average_degree"])
        plt.title("Average Degree Over Time")
        plt.xlabel("Step")
        plt.ylabel("Average Degree")
        plt.show()

        # Revolt Frequency
        plt.figure()
        plt.bar(range(steps), self.metrics["revolt_frequency"])
        plt.title("Revolt Frequency Over Time")
        plt.xlabel("Step")
        plt.ylabel("Revolt Count")
        plt.show()

        # Opinion Distribution
        plt.figure()
        final_opinions = self.metrics["opinion_distribution"][-1]
        plt.hist(final_opinions, bins=20, edgecolor="black")
        plt.title("Final Opinion Distribution")
        plt.xlabel("Opinion")
        plt.ylabel("Frequency")
        plt.show()


    def calculate_polarization_index(self):
        opinions = np.array([a.opinion for a in self.schedule.agents if isinstance(a, IndividualAgent)])
        return np.mean(np.abs(opinions))  # Example polarization index

    def print_hub_connections(self):
        for agent in self.schedule.agents:
            if "Propaganda Hub" in agent.name:  # Check if the agent is a Propaganda Hub
                neighbors = list(graph.neighbors(agent.unique_id))
                print(f"{agent.name} is connected to: {neighbors}")
```


```python
# def visualize_data(model):
#     fig = make_subplots(rows=5, cols=1,  # 5 rows for 5 plots
#                         specs=[[{"type": "histogram"}],
#                                [{"type": "scatter"}],
#                                [{"type": "scatter"}],
#                                [{"type": "bar"}],
#                                [{"type": "scatter"}]],  # Assuming scatter for network dynamics
#                         subplot_titles=("Opinion Distribution", "Polarization Index",
#                                         "Average Opinion Change", "Revolt Frequency",
#                                         "Network Dynamics"))

#     # Opinion Distribution
#     fig.add_trace(go.Histogram(x=model.opinion_distribution[-1], nbinsx=20), row=1, col=1)

#     # Polarization Index
#     fig.add_trace(go.Scatter(x=list(range(len(model.polarization_index))), y=model.polarization_index), row=2, col=1)

#     # Average Opinion Change
#     fig.add_trace(go.Scatter(x=list(range(len(model.avg_opinion_change))), y=model.avg_opinion_change), row=3, col=1)

#     # Revolt Frequency
#     fig.add_trace(go.Bar(x=list(range(len(model.revolt_count))), y=model.revolt_count), row=4, col=1)

#     # Network Dynamics (using a scatter plot as a placeholder)
#     degrees = [degree for node, degree in model.network.degree()]
#     fig.add_trace(go.Scatter(x=list(range(len(degrees))), y=degrees, mode='markers'), row=5, col=1)  # Scatter plot for network dynamics

#     fig.update_layout(height=1200, width=1000, title_text="ABM Simulation Results")  # Adjust height as needed
#     fig.show()
```


```python
def initialize_ABM():
    global starter_model
    n_individuals=1000
    n_propaganda_hubs=50
    starter_model = ExperimentModel(n_individuals, n_propaganda_hubs)

def run_abm_step():
    starter_model.step()  # Execute one step of the ABM

def run_abm_sayhi():
    # starter_model.print_hub_connections()  # Call the function here
    starter_model.agents.shuffle_do("say_hi")  # Execute one step of the ABM

ABM_init_time = timeit.timeit(initialize_ABM, number=1)  # Run once
print(f"ABM init time: {ABM_init_time} seconds")
sayhi_execution_time = timeit.timeit(run_abm_sayhi, number=1)  # Run once
print(f"Say hi execution time: {sayhi_execution_time} seconds")
```

    ABM init time: 0.14338544900238048 seconds
    Hi, I am Individual 832 with opinion 0.2716365761268129, resistance 0.6319603946176575, influence 1.2886733767961271, and revolt threshold 0.9997642972306808
    Hi, I am Individual 374 with opinion -0.5184401566272843, resistance 0.10216924175690123, influence 0.8186362605832546, and revolt threshold 0.8654576586029914
    Hi, I am Individual 521 with opinion -0.6581325788029115, resistance 0.8496747566952596, influence 0.847482806813519, and revolt threshold 0.9073288680183746
    Hi, I am Individual 670 with opinion 0.6989783385066177, resistance 0.7871784763603186, influence 1.4040485242445673, and revolt threshold 0.9332963773263732
    Hi, I am Individual 362 with opinion 0.4511463298699234, resistance 0.056505559509488434, influence 1.4738563459034024, and revolt threshold 0.9610869016780402
    Hi, I am Individual 101 with opinion -0.297595165225895, resistance 0.6954854444490388, influence 0.9026575379438225, and revolt threshold 0.792948686068103
    Hi, I am Individual 71 with opinion -0.9092962895369503, resistance 0.36155608679948636, influence 0.9858480205344077, and revolt threshold 0.9621757054798512
    Hi, I am Individual 573 with opinion -0.3620984081292282, resistance 0.09395698468893587, influence 1.1289060406676272, and revolt threshold 0.8747909236763494
    Hi, I am Individual 37 with opinion -0.033679870700187964, resistance 0.12290597632256894, influence 0.9089333991133549, and revolt threshold 0.8427214338534239
    Hi, I am Individual 522 with opinion -0.2957446781692534, resistance 0.6259785778524294, influence 0.9583607880728272, and revolt threshold 0.9076949358676892
    Hi, I am Individual 843 with opinion -0.5496652041026244, resistance 0.23983832658116166, influence 0.9578878419738139, and revolt threshold 0.9241898473065626
    Hi, I am Individual 277 with opinion -0.6903206107696693, resistance 0.9531034993353875, influence 1.138249597411451, and revolt threshold 0.9359199483177578
    Hi, I am Individual 600 with opinion 0.35718917873186706, resistance 0.28513247925336527, influence 1.3049983474421953, and revolt threshold 0.8464656500000044
    Hi, I am Individual 332 with opinion -0.3495844662810985, resistance 0.7736735697300224, influence 1.0482643450311395, and revolt threshold 0.9235985984751468
    Hi, I am Individual 349 with opinion 0.8690625899486519, resistance 0.8854286684360202, influence 1.048317993957892, and revolt threshold 0.9093438234963799
    Hi, I am Individual 448 with opinion -0.6919478009040081, resistance 0.2315248158569405, influence 0.9232798656908148, and revolt threshold 0.8917366418420891
    Hi, I am Individual 483 with opinion -0.9025204746545439, resistance 0.9803539686416947, influence 0.8714979980508819, and revolt threshold 0.8534388224646459
    Hi, I am Individual 405 with opinion 0.7106283821613713, resistance 0.44032415164513106, influence 1.024697227789079, and revolt threshold 0.9269100908693366
    Hi, I am Individual 411 with opinion 0.4258442170633914, resistance 0.8150225497511351, influence 1.3837764221984972, and revolt threshold 0.9610249724623223
    Hi, I am Individual 897 with opinion -0.2518099109320495, resistance 0.5469704557028665, influence 0.8767222776919923, and revolt threshold 0.8305971155042294
    Hi, I am Individual 824 with opinion 0.2686177395754774, resistance 0.019905645399622474, influence 0.8338481834474356, and revolt threshold 0.966369355173402
    Hi, I am Individual 750 with opinion -0.35188657765330245, resistance 0.30733163961689747, influence 1.29231853651461, and revolt threshold 0.8879106688785937
    Hi, I am Individual 590 with opinion -0.8763535003725342, resistance 0.8209230103245209, influence 1.4876232088828556, and revolt threshold 0.9877253811405564
    Hi, I am Individual 288 with opinion -0.5113198236512992, resistance 0.00539107806315009, influence 0.9949094338409115, and revolt threshold 0.8680062040728325
    Hi, I am Individual 281 with opinion 0.1374083450297361, resistance 0.4625241214261139, influence 1.0693481364873956, and revolt threshold 0.904098297222725
    Hi, I am Individual 84 with opinion 0.8704688028155523, resistance 0.8416038093872877, influence 1.250849792226072, and revolt threshold 0.9547531622977794
    Hi, I am Individual 633 with opinion -0.28186351129180465, resistance 0.43939394032734314, influence 0.8257740810324881, and revolt threshold 0.8368501523448786
    Hi, I am Individual 681 with opinion -0.1402478244348646, resistance 0.38372487910459707, influence 1.24531429292532, and revolt threshold 0.9341806042818116
    Hi, I am Individual 121 with opinion -0.21028683002997073, resistance 0.8821941915777012, influence 1.3945178349098262, and revolt threshold 0.7060148762672653
    Hi, I am Individual 543 with opinion 0.935427281429579, resistance 0.11821114906244234, influence 1.2108910085769264, and revolt threshold 0.8818015197982157
    Hi, I am Individual 655 with opinion -0.6757756590693389, resistance 0.3764316164535384, influence 0.9841424320988645, and revolt threshold 0.7715110401686527
    Hi, I am Individual 211 with opinion -0.8877583065443164, resistance 0.14991829012227587, influence 1.343307926658381, and revolt threshold 0.9410945557359989
    Hi, I am Individual 983 with opinion -0.6745673917419812, resistance 0.8477835644820657, influence 1.380254772675761, and revolt threshold 0.9511757923545926
    Hi, I am Individual 369 with opinion 0.963004880581714, resistance 0.6341597930313775, influence 1.3741595986794213, and revolt threshold 0.9754899827649368
    Hi, I am Individual 450 with opinion 0.33918696365492074, resistance 0.02413196596400924, influence 1.2052389138627908, and revolt threshold 0.813231416395895
    Hi, I am Individual 892 with opinion -0.21293461574384498, resistance 0.12731351669402824, influence 1.1069905179927257, and revolt threshold 0.7697349225755548
    Hi, I am Individual 693 with opinion 0.6530300609684982, resistance 0.3501328811622254, influence 1.392208905268951, and revolt threshold 0.74804748973528
    Hi, I am Individual 178 with opinion -0.6156036766996866, resistance 0.07369157458067088, influence 0.909214149845582, and revolt threshold 0.9550515119589951
    Hi, I am an propaganda agent, you can call me Propaganda Hub 1028 and I have the id of 1028.       I have the opinion of -1, and influence strength is 0.5416625863927851      with a radius of 56.936650926501464.
    Hi, I am Individual 168 with opinion -0.9068115296582187, resistance 0.02830614170596557, influence 1.442540948134405, and revolt threshold 0.8744438090530593
    Hi, I am an propaganda agent, you can call me Propaganda Hub 1016 and I have the id of 1016.       I have the opinion of 1, and influence strength is 0.7316367665025976      with a radius of 18.678126402861253.
    Hi, I am Individual 520 with opinion -0.8050732336621793, resistance 0.3570808818682627, influence 1.0001679390884863, and revolt threshold 0.8975665607019996
    Hi, I am Individual 146 with opinion 0.7350681131338517, resistance 0.09487484999734774, influence 1.052898659592186, and revolt threshold 0.988404401989164
    Hi, I am Individual 200 with opinion -0.21715528416614394, resistance 0.09537463214364783, influence 1.3586827619650645, and revolt threshold 0.7978178121720731
    Hi, I am Individual 305 with opinion -0.7085150463680876, resistance 0.4413602483192277, influence 1.0519403923805142, and revolt threshold 0.7959119013840964
    Hi, I am Individual 96 with opinion -0.5563324604481952, resistance 0.3048568818763604, influence 1.110035357829661, and revolt threshold 0.7290174735171032
    Hi, I am Individual 851 with opinion -0.08525637556978083, resistance 0.4288165720035486, influence 0.8787775406122361, and revolt threshold 0.8678087341073804
    Hi, I am Individual 746 with opinion 0.3446905823541877, resistance 0.23056479992202228, influence 1.0782926325476565, and revolt threshold 0.7728726922632887
    Hi, I am Individual 637 with opinion -0.2446653066560578, resistance 0.9490166479550397, influence 1.1741534857160447, and revolt threshold 0.7864845044492632
    Hi, I am Individual 928 with opinion 0.3974205562444513, resistance 0.7070521098104366, influence 1.0356290909506918, and revolt threshold 0.957982088412569
    Hi, I am Individual 929 with opinion 0.019980793605953506, resistance 0.1355545933849941, influence 0.9146845448270646, and revolt threshold 0.9971533028029391
    Hi, I am Individual 660 with opinion -0.997011805335102, resistance 0.5137048186116082, influence 1.0069167286397782, and revolt threshold 0.8905415390885277
    Hi, I am Individual 764 with opinion 0.01148657998711955, resistance 0.6071030773257502, influence 1.3678750212466166, and revolt threshold 0.8880097244283838
    Hi, I am Individual 256 with opinion 0.5431002096845974, resistance 0.9134614303488079, influence 1.2082736330996944, and revolt threshold 0.7290661199997642
    Hi, I am Individual 728 with opinion 0.5547914557266935, resistance 0.4492791249153879, influence 1.450964517479715, and revolt threshold 0.8677186366605638
    Hi, I am Individual 260 with opinion -0.6548642871689165, resistance 0.16371937844888806, influence 1.2262036242879168, and revolt threshold 0.8127652008513297
    Hi, I am Individual 91 with opinion 0.3829672472854577, resistance 0.8131038756199498, influence 1.3809547739541397, and revolt threshold 0.9036200306764707
    Hi, I am Individual 979 with opinion 0.35953963774100406, resistance 0.5444768248259202, influence 1.3065046191104466, and revolt threshold 0.8837857881373424
    Hi, I am Individual 993 with opinion 0.9390701584453431, resistance 0.7476123168623132, influence 1.0101295292291221, and revolt threshold 0.9632432479426863
    Hi, I am Individual 286 with opinion 0.2977370841393674, resistance 0.08234238740911726, influence 1.0993698824810414, and revolt threshold 0.8628049233713999
    Hi, I am Individual 385 with opinion -0.522411743851172, resistance 0.3968897632529872, influence 1.1700249688731508, and revolt threshold 0.8111059177787646
    Hi, I am Individual 752 with opinion 0.5942812171364813, resistance 0.8857161000198073, influence 1.4142409475617075, and revolt threshold 0.7193403595747445
    Hi, I am Individual 795 with opinion -0.9100142533764237, resistance 0.6240555044734704, influence 1.2718251791146669, and revolt threshold 0.8076113264533697
    Hi, I am Individual 659 with opinion 0.11903006135364791, resistance 0.5410738668695385, influence 1.0016437190203338, and revolt threshold 0.9227286788570139
    Hi, I am Individual 886 with opinion -0.9809107364511354, resistance 0.1319125207651023, influence 1.2229384173629945, and revolt threshold 0.8476745069507357
    Hi, I am Individual 918 with opinion -0.5727908972981912, resistance 0.8092693771297914, influence 1.1101676944817713, and revolt threshold 0.8062626403147755
    Hi, I am Individual 731 with opinion -0.9201657401133452, resistance 0.4471822942413817, influence 1.3875343192042, and revolt threshold 0.9765026804094069
    Hi, I am Individual 900 with opinion -0.6323026888515246, resistance 0.6159303487420233, influence 1.2228301048925807, and revolt threshold 0.8512537249949573
    Hi, I am Individual 384 with opinion 0.8824275262883403, resistance 0.12329720877961892, influence 1.47845002798481, and revolt threshold 0.832389506210445
    Hi, I am Individual 343 with opinion 0.43006500866883957, resistance 0.35717588059283756, influence 0.9314818948876862, and revolt threshold 0.8051185520557891
    Hi, I am Individual 887 with opinion 0.16780270529121455, resistance 0.07343802765196061, influence 1.3984975476284227, and revolt threshold 0.735610522440746
    Hi, I am Individual 558 with opinion 0.4114887668045333, resistance 0.09033059185678838, influence 0.9436498952540147, and revolt threshold 0.7305988573211742
    Hi, I am Individual 503 with opinion 0.6600151588878711, resistance 0.18847805373046822, influence 1.245720096100762, and revolt threshold 0.7250740578631147
    Hi, I am Individual 873 with opinion 0.9850270621694066, resistance 0.16568397655472, influence 1.4874714154553859, and revolt threshold 0.7902969867883785
    Hi, I am Individual 364 with opinion -0.8227430712433124, resistance 0.5506935694957231, influence 1.4714023522521393, and revolt threshold 0.7673014916698377
    Hi, I am Individual 703 with opinion -0.19556601024673026, resistance 0.11895279028850458, influence 0.9566452109805387, and revolt threshold 0.8299932910584564
    Hi, I am Individual 115 with opinion 0.3482233464109783, resistance 0.17463882934398922, influence 1.3068935640082504, and revolt threshold 0.7535864683385557
    Hi, I am Individual 582 with opinion -0.8208478285978331, resistance 0.8283793651664401, influence 1.4744007833440333, and revolt threshold 0.8603219309775125
    Hi, I am Individual 945 with opinion -0.6256451509392509, resistance 0.031480621262160446, influence 1.3768667478129943, and revolt threshold 0.7605031272363255
    Hi, I am Individual 62 with opinion 0.32701599138325843, resistance 0.606869442261024, influence 1.3002326290050124, and revolt threshold 0.727138174063676
    Hi, I am Individual 32 with opinion -0.9660386964648191, resistance 0.5902659519860498, influence 1.1104656108378026, and revolt threshold 0.9521677937771666
    Hi, I am Individual 381 with opinion 0.6973354990712342, resistance 0.9606918692267757, influence 1.414031639353213, and revolt threshold 0.8497461338492756
    Hi, I am Individual 280 with opinion 0.1449462629313374, resistance 0.046181697857586745, influence 1.4345221671275705, and revolt threshold 0.9376575239095013
    Hi, I am Individual 663 with opinion 0.30005480165726883, resistance 0.05409392687555503, influence 1.1580856626701812, and revolt threshold 0.8229389744173695
    Hi, I am Individual 713 with opinion -0.8415725995060999, resistance 0.55501388147782, influence 1.1578456922993887, and revolt threshold 0.8120719077553032
    Hi, I am Individual 837 with opinion 0.41873350604173565, resistance 0.7928752609249984, influence 0.816345696154764, and revolt threshold 0.8158299872614717
    Hi, I am an propaganda agent, you can call me Propaganda Hub 1031 and I have the id of 1031.       I have the opinion of -1, and influence strength is 0.5678552157739158      with a radius of 45.07620118115514.
    Hi, I am Individual 191 with opinion -0.9325501098718785, resistance 0.48284670502253524, influence 0.8816845233638414, and revolt threshold 0.9306174653062305
    Hi, I am Individual 33 with opinion -0.20077079980898827, resistance 0.24752680220269274, influence 0.8800542885821283, and revolt threshold 0.7314469833198997
    Hi, I am Individual 113 with opinion -0.5894558089663038, resistance 0.33786208915802196, influence 1.2319023604337735, and revolt threshold 0.9862449365191757
    Hi, I am Individual 419 with opinion 0.20519795170344302, resistance 0.27510782725645266, influence 1.4153737481242574, and revolt threshold 0.9668949087146904
    Hi, I am Individual 787 with opinion 0.18245479579746582, resistance 0.23612561871552595, influence 1.3019832407405636, and revolt threshold 0.8993439694123363
    Hi, I am Individual 687 with opinion -0.5207753763729064, resistance 0.7077203412306908, influence 1.3890798974619862, and revolt threshold 0.7994953834997545
    Hi, I am Individual 668 with opinion -0.5521265045309327, resistance 0.8303419206713434, influence 1.391856052912134, and revolt threshold 0.780315628984712
    Hi, I am Individual 692 with opinion 0.2813858892215364, resistance 0.30955926153123947, influence 1.2364945013741941, and revolt threshold 0.8387765143178731
    Hi, I am Individual 251 with opinion 0.4266179231243705, resistance 0.823754610663013, influence 0.9403011983947411, and revolt threshold 0.8961941667435654
    Hi, I am Individual 409 with opinion -0.37579960256176825, resistance 0.9964277461839756, influence 1.413386603026744, and revolt threshold 0.7623729678581063
    Hi, I am Individual 557 with opinion 0.040572216635602176, resistance 0.4418562635603597, influence 1.0464208099707117, and revolt threshold 0.9184461267845088
    Hi, I am Individual 973 with opinion 0.2648754936997191, resistance 0.9087186108003747, influence 1.3593193881588033, and revolt threshold 0.7553706581272039
    Hi, I am Individual 544 with opinion -0.9983059100372322, resistance 0.6072023179669049, influence 1.407562258823432, and revolt threshold 0.7583092994960349
    Hi, I am Individual 940 with opinion 0.8689615353206463, resistance 0.4711783783725504, influence 0.9625422861156866, and revolt threshold 0.8732157152166297
    Hi, I am Individual 878 with opinion -0.16012816887780068, resistance 0.017535478120598857, influence 1.2681903041922187, and revolt threshold 0.7194355218585734
    Hi, I am Individual 23 with opinion -0.2126679355209813, resistance 0.48474291013002946, influence 1.210523203831026, and revolt threshold 0.8368735158331717
    Hi, I am Individual 75 with opinion 0.7702725553096861, resistance 0.37951751151803326, influence 1.1651045642431608, and revolt threshold 0.9654089575765886
    Hi, I am Individual 318 with opinion 0.3276409536443703, resistance 0.16085507510354513, influence 0.8440520849806602, and revolt threshold 0.709981575654455
    Hi, I am Individual 476 with opinion -0.3763906430012649, resistance 0.7352155518110398, influence 1.1254102723764303, and revolt threshold 0.7379697734374165
    Hi, I am Individual 11 with opinion -0.27617276375989497, resistance 0.47361615525474277, influence 1.1307114896806905, and revolt threshold 0.9192472048671033
    Hi, I am Individual 195 with opinion 0.4668821433233601, resistance 0.6923128245779158, influence 0.8148964203381902, and revolt threshold 0.9795666460524389
    Hi, I am Individual 201 with opinion 0.674708752930445, resistance 0.8296414139809909, influence 1.222353173695005, and revolt threshold 0.9059366096771235
    Hi, I am Individual 792 with opinion -0.2803695553675509, resistance 0.6432137700315701, influence 1.2815259519378048, and revolt threshold 0.965988355022247
    Hi, I am Individual 898 with opinion -0.5610952180255953, resistance 0.3469255783510715, influence 0.996552938872491, and revolt threshold 0.7233645815550411
    Hi, I am Individual 436 with opinion 0.18007617932270148, resistance 0.36642631154005667, influence 1.0640014728663445, and revolt threshold 0.8566409424380056
    Hi, I am Individual 415 with opinion -0.9319996514378965, resistance 0.21946659115957767, influence 1.3930272048947998, and revolt threshold 0.9481807967484286
    Hi, I am Individual 501 with opinion 0.06731088805506591, resistance 0.36861561846058477, influence 1.060508091378054, and revolt threshold 0.9385933111284224
    Hi, I am Individual 294 with opinion -0.45320220541618594, resistance 0.26259773338941206, influence 1.1118929338950865, and revolt threshold 0.8083643502857549
    Hi, I am Individual 533 with opinion -0.44321135741545326, resistance 0.4476674860603155, influence 1.3962357394895701, and revolt threshold 0.9756328434221564
    Hi, I am Individual 306 with opinion -0.9863419795550656, resistance 0.6479633842718983, influence 1.1166873123173788, and revolt threshold 0.9647570682076347
    Hi, I am Individual 111 with opinion 0.626900590891561, resistance 0.8945244625495534, influence 0.9806659959192766, and revolt threshold 0.8568973694938686
    Hi, I am Individual 63 with opinion -0.8545687338526984, resistance 0.3856297472320145, influence 1.4390630006681835, and revolt threshold 0.8115239257548855
    Hi, I am Individual 48 with opinion -0.7543620032024418, resistance 0.2648409879869068, influence 1.216779907845393, and revolt threshold 0.8425348432688563
    Hi, I am Individual 563 with opinion -0.3945425104565421, resistance 0.7721767058391429, influence 0.9429431072504229, and revolt threshold 0.8347187052373068
    Hi, I am Individual 575 with opinion -0.614185580681196, resistance 0.9088242258174941, influence 1.2976258324194805, and revolt threshold 0.811082100846318
    Hi, I am Individual 591 with opinion 0.1975206792982127, resistance 0.2562500052807354, influence 0.8928630455210594, and revolt threshold 0.8288845066175907
    Hi, I am Individual 763 with opinion -0.7667472423132862, resistance 0.06201223688450863, influence 1.2585843046757215, and revolt threshold 0.808758570530692
    Hi, I am Individual 818 with opinion -0.9663972773796154, resistance 0.3013219041873352, influence 1.2431874816986968, and revolt threshold 0.7554630227881232
    Hi, I am an propaganda agent, you can call me Propaganda Hub 1026 and I have the id of 1026.       I have the opinion of -1, and influence strength is 0.8927031001560556      with a radius of 33.424920071782395.
    Hi, I am an propaganda agent, you can call me Propaganda Hub 1042 and I have the id of 1042.       I have the opinion of 1, and influence strength is 0.49635491706410834      with a radius of 28.66315103212425.
    Hi, I am Individual 893 with opinion 0.5180642531698267, resistance 0.7170219387664235, influence 1.4532894234802514, and revolt threshold 0.9942455364069533
    Hi, I am Individual 163 with opinion -0.027169541152345378, resistance 0.17209229594642783, influence 0.9955707672638768, and revolt threshold 0.8410969638141728
    Hi, I am Individual 721 with opinion -0.5843883905835376, resistance 0.2472072182392756, influence 0.8974602512054131, and revolt threshold 0.8993659530104652
    Hi, I am Individual 93 with opinion 0.45680443972185514, resistance 0.3999187552216832, influence 0.8881813064166874, and revolt threshold 0.8249618722796477
    Hi, I am Individual 16 with opinion 0.45190196259496895, resistance 0.8814117163665195, influence 1.2044074590349825, and revolt threshold 0.8575287496284458
    Hi, I am Individual 403 with opinion 0.29823933201007, resistance 0.645168637320289, influence 1.190697015920763, and revolt threshold 0.9642715119321252
    Hi, I am Individual 97 with opinion 0.0038683975215401123, resistance 0.9995891251779708, influence 1.2554819569081372, and revolt threshold 0.7680731372134604
    Hi, I am Individual 944 with opinion -0.3050545631269115, resistance 0.25266535351453756, influence 1.4266860553058862, and revolt threshold 0.78190305525651
    Hi, I am Individual 441 with opinion -0.7123523557496905, resistance 0.6398608837069022, influence 1.0018240513859935, and revolt threshold 0.9516714943675983
    Hi, I am Individual 132 with opinion 0.9237856912028382, resistance 0.9221706408590735, influence 0.8027712742202882, and revolt threshold 0.75577491941645
    Hi, I am Individual 406 with opinion 0.23767142989029133, resistance 0.687622134305551, influence 0.88823044059771, and revolt threshold 0.9543062985831998
    Hi, I am Individual 673 with opinion -0.7954079102769249, resistance 0.07537392990999181, influence 0.9288393236477159, and revolt threshold 0.9727674984381576
    Hi, I am Individual 786 with opinion 0.29733820927755605, resistance 0.9213744266669595, influence 1.1456213792877525, and revolt threshold 0.9009072154684826
    Hi, I am Individual 538 with opinion 0.9644103586850894, resistance 0.31470597253166976, influence 0.9037912174392623, and revolt threshold 0.9420630721723968
    Hi, I am Individual 497 with opinion -0.3550803549818473, resistance 0.5583945325221155, influence 1.0533428930596562, and revolt threshold 0.7413723506069109
    Hi, I am Individual 174 with opinion 0.337200076467286, resistance 0.3577249314181288, influence 1.180600622886022, and revolt threshold 0.8342850678158129
    Hi, I am Individual 628 with opinion -0.4206719060902422, resistance 0.9701843234928103, influence 1.059612948794267, and revolt threshold 0.7959736718669299
    Hi, I am Individual 654 with opinion 0.4113336655635804, resistance 0.3302735892227152, influence 0.813983389043272, and revolt threshold 0.9241922898858582
    Hi, I am Individual 570 with opinion 0.31722432943179024, resistance 0.1062580119583072, influence 1.064630196593055, and revolt threshold 0.7949968315790665
    Hi, I am Individual 140 with opinion -0.5917614150120254, resistance 0.5311925727523511, influence 1.1966280647844545, and revolt threshold 0.7160608314074043
    Hi, I am Individual 632 with opinion 0.6120602012180503, resistance 0.926091786001831, influence 0.8358272050459863, and revolt threshold 0.9203849128730766
    Hi, I am Individual 90 with opinion 0.7789481035081716, resistance 0.18926834217914357, influence 1.13332358535786, and revolt threshold 0.9710109544430912
    Hi, I am Individual 598 with opinion -0.811447695354496, resistance 0.036204572622832165, influence 1.085010975103561, and revolt threshold 0.8547709930903385
    Hi, I am Individual 990 with opinion 0.011171132210227697, resistance 0.34174287407617177, influence 1.3290811822738138, and revolt threshold 0.7210599405048433
    Hi, I am Individual 932 with opinion 0.8315611892059398, resistance 0.8337575618601992, influence 1.3617719283417764, and revolt threshold 0.8948055342851224
    Hi, I am Individual 949 with opinion -0.675433527553237, resistance 0.5735346810811407, influence 0.9194659459740766, and revolt threshold 0.7141512106152255
    Hi, I am Individual 995 with opinion -0.4119516807208736, resistance 0.6283067358562654, influence 1.249147538142726, and revolt threshold 0.7249160959757565
    Hi, I am Individual 297 with opinion 0.22750315137625376, resistance 0.34036204364069145, influence 0.8145537706409631, and revolt threshold 0.9269188965696789
    Hi, I am Individual 329 with opinion -0.9730151653261814, resistance 0.3309655819063747, influence 1.3718702537860559, and revolt threshold 0.885513054417922
    Hi, I am Individual 391 with opinion 0.23288818654754606, resistance 0.07229938979752992, influence 1.1946675499401962, and revolt threshold 0.8549192200474247
    Hi, I am Individual 117 with opinion 0.28568883328195227, resistance 0.21531175997257845, influence 1.0025183224428538, and revolt threshold 0.7408662319088317
    Hi, I am an propaganda agent, you can call me Propaganda Hub 1015 and I have the id of 1015.       I have the opinion of 1, and influence strength is 0.6506700263172217      with a radius of 57.73220695185323.
    Hi, I am Individual 896 with opinion -0.42873635910117325, resistance 0.71345638758693, influence 0.9547936171761615, and revolt threshold 0.8613223540544512
    Hi, I am Individual 431 with opinion 0.9768595331828791, resistance 0.5999562209785915, influence 1.0491430685328642, and revolt threshold 0.874202376963326
    Hi, I am Individual 296 with opinion -0.2576436512799838, resistance 0.18256553077618487, influence 0.9697991118474728, and revolt threshold 0.9104329592025113
    Hi, I am Individual 565 with opinion 0.6204776287076164, resistance 0.07694379223637826, influence 1.1413882602164456, and revolt threshold 0.807618593749997
    Hi, I am Individual 939 with opinion 0.37562913958841126, resistance 0.43360156576407016, influence 1.3536608213151613, and revolt threshold 0.8696339511718011
    Hi, I am Individual 691 with opinion -0.8927975762495233, resistance 0.006155759740033573, influence 0.8248071919012252, and revolt threshold 0.9009819283439449
    Hi, I am Individual 500 with opinion -0.13434743801652393, resistance 0.6174759236197775, influence 1.0355403393551157, and revolt threshold 0.7068993718358461
    Hi, I am Individual 797 with opinion -0.46996767066734413, resistance 0.6669550863675482, influence 1.06706829191771, and revolt threshold 0.8326162616505739
    Hi, I am Individual 796 with opinion 0.33723697968292843, resistance 0.24741214109457332, influence 1.2173517237064315, and revolt threshold 0.9299352040140634
    Hi, I am Individual 443 with opinion 0.9264356788312915, resistance 0.5333705668560805, influence 0.8517868512116276, and revolt threshold 0.9871859896327603
    Hi, I am Individual 552 with opinion 0.4364493013230264, resistance 0.04403574640328212, influence 0.8710617483785874, and revolt threshold 0.8775421337837702
    Hi, I am Individual 650 with opinion 0.6210890813979904, resistance 0.6058175212047602, influence 1.3603637424935922, and revolt threshold 0.7327628170285018
    Hi, I am Individual 577 with opinion -0.4329876902505554, resistance 0.6352541930023035, influence 0.884327623271962, and revolt threshold 0.8840050614384157
    Hi, I am Individual 363 with opinion -0.7688428869613266, resistance 0.9014721247026045, influence 1.4989257655265185, and revolt threshold 0.7542615670152879
    Hi, I am Individual 921 with opinion 0.6456130414820089, resistance 0.01307427371416503, influence 0.8829645948629247, and revolt threshold 0.975186369745032
    Hi, I am Individual 718 with opinion 0.6935164404355418, resistance 0.2296960699500754, influence 1.3448860822591893, and revolt threshold 0.7638251703272642
    Hi, I am Individual 583 with opinion -0.8443085807645025, resistance 0.7622880453613553, influence 1.2723215697663495, and revolt threshold 0.7286518784491576
    Hi, I am Individual 608 with opinion 0.8844481440340299, resistance 0.468274909816467, influence 1.4341862444542848, and revolt threshold 0.9455023712997765
    Hi, I am Individual 838 with opinion 0.026295291990851455, resistance 0.6136819091729225, influence 1.0360179536054746, and revolt threshold 0.9599370275094611
    Hi, I am Individual 310 with opinion -0.18315755494273533, resistance 0.948393403880798, influence 1.1502798082650916, and revolt threshold 0.8632444155656793
    Hi, I am Individual 749 with opinion -0.846610269249545, resistance 0.9494515032676584, influence 1.4747501227220439, and revolt threshold 0.8016753719622123
    Hi, I am Individual 354 with opinion 0.12850752616592875, resistance 0.4736983854151319, influence 1.3035918568335365, and revolt threshold 0.9899460886550737
    Hi, I am Individual 616 with opinion -0.318094982017896, resistance 0.5644017769409923, influence 1.014883178817256, and revolt threshold 0.9749764846234993
    Hi, I am Individual 742 with opinion 0.8538041920086021, resistance 0.2885531675969871, influence 0.8658133548589632, and revolt threshold 0.874383741561604
    Hi, I am Individual 437 with opinion 0.7261369850799437, resistance 0.5210362219577876, influence 1.0275474680399639, and revolt threshold 0.8484863362748765
    Hi, I am Individual 455 with opinion -0.6914719668677434, resistance 0.5760863130078936, influence 1.3683936609913319, and revolt threshold 0.805154892291994
    Hi, I am Individual 701 with opinion 0.46527955062578674, resistance 0.8884523272962171, influence 0.9027075997146659, and revolt threshold 0.9093873327895363
    Hi, I am Individual 159 with opinion -0.13284246689808477, resistance 0.1341719102049509, influence 1.4442212769963578, and revolt threshold 0.7292238907342053
    Hi, I am Individual 371 with opinion -0.9772531861428095, resistance 0.726355272051123, influence 0.9053248244352293, and revolt threshold 0.7865750970950052
    Hi, I am Individual 184 with opinion -0.5482819602356002, resistance 0.8268995692315448, influence 1.0216684251101684, and revolt threshold 0.9127894215102161
    Hi, I am an propaganda agent, you can call me Propaganda Hub 1045 and I have the id of 1045.       I have the opinion of -1, and influence strength is 0.8763188882203475      with a radius of 55.365080867240735.
    Hi, I am Individual 116 with opinion 0.18411403495208378, resistance 0.19281072606062644, influence 0.9157542451086156, and revolt threshold 0.9006027578921445
    Hi, I am Individual 56 with opinion 0.7655683263875375, resistance 0.05248271136798799, influence 1.083654447876746, and revolt threshold 0.9245157632649273
    Hi, I am Individual 55 with opinion 0.32878114749371234, resistance 0.978065590672292, influence 1.2932902613869532, and revolt threshold 0.7221204107588283
    Hi, I am Individual 333 with opinion 0.5006115707859662, resistance 0.29816609946645223, influence 1.3187321186682324, and revolt threshold 0.9322858691597268
    Hi, I am Individual 377 with opinion 0.6524619833418743, resistance 0.8702937186589361, influence 0.9581599723346661, and revolt threshold 0.8372766648515667
    Hi, I am Individual 45 with opinion -0.5435125852129941, resistance 0.3236659072343958, influence 1.4290563258501199, and revolt threshold 0.8485734945148948
    Hi, I am Individual 911 with opinion 0.42938908860932856, resistance 0.2960274278730223, influence 1.1129765990366074, and revolt threshold 0.7818124386909563
    Hi, I am Individual 761 with opinion 0.34808919173312836, resistance 0.13895230651310764, influence 1.4140350641492048, and revolt threshold 0.7547187408056215
    Hi, I am Individual 639 with opinion -0.4506889965823504, resistance 0.38101610218398996, influence 1.1924661206201757, and revolt threshold 0.9671162422965749
    Hi, I am Individual 479 with opinion 0.5596318866381287, resistance 0.013012317570141652, influence 1.0373210829676722, and revolt threshold 0.959983126837648
    Hi, I am Individual 536 with opinion -0.36063133352405785, resistance 0.34981045099958785, influence 1.2823561112333017, and revolt threshold 0.79370661802578
    Hi, I am Individual 92 with opinion 0.10575667446614045, resistance 0.23864636728559208, influence 1.260000056792765, and revolt threshold 0.86826738613841
    Hi, I am Individual 489 with opinion 0.45415718028756036, resistance 0.9972983998210109, influence 1.0739890759265371, and revolt threshold 0.7508089833895635
    Hi, I am Individual 546 with opinion 0.011036476584194599, resistance 0.19968547342156873, influence 0.9802234099323639, and revolt threshold 0.9579966078267439
    Hi, I am Individual 496 with opinion 0.25988717379811055, resistance 0.08652733692997705, influence 1.1685766198815069, and revolt threshold 0.8768102114266747
    Hi, I am Individual 775 with opinion 0.7012718491254699, resistance 0.3926667649999449, influence 1.0500278798856026, and revolt threshold 0.8590393594296569
    Hi, I am Individual 817 with opinion -0.7140734763467067, resistance 0.4933751698800778, influence 0.9495802880060205, and revolt threshold 0.7292390750768856
    Hi, I am Individual 950 with opinion 0.7712464941239869, resistance 0.8494811187712261, influence 0.9428498438869006, and revolt threshold 0.8077691453646647
    Hi, I am Individual 358 with opinion -0.8459347321073194, resistance 0.8536635372754184, influence 0.8863783914670658, and revolt threshold 0.7415542808357537
    Hi, I am Individual 946 with opinion 0.7322427399297915, resistance 0.4998032658998154, influence 0.9752359775557627, and revolt threshold 0.8182029802059829
    Hi, I am Individual 738 with opinion 0.8567369466264245, resistance 0.3228682185256303, influence 0.9255842153080162, and revolt threshold 0.7169550300923611
    Hi, I am Individual 585 with opinion -0.8867093950790987, resistance 0.37661486190907245, influence 1.4035102758285642, and revolt threshold 0.9189486239314684
    Hi, I am Individual 57 with opinion -0.4845936033920386, resistance 0.28375860005555753, influence 1.2555942462184242, and revolt threshold 0.8840279990715461
    Hi, I am Individual 107 with opinion 0.24268362763822227, resistance 0.36078611419870754, influence 1.1763982223597946, and revolt threshold 0.9863990688653757
    Hi, I am Individual 744 with opinion -0.8155061381379438, resistance 0.21836456218804545, influence 0.9490563357710863, and revolt threshold 0.8825718667895525
    Hi, I am Individual 19 with opinion -0.07539255442525938, resistance 0.7551382225053824, influence 0.9426211897391594, and revolt threshold 0.8891605154634803
    Hi, I am Individual 408 with opinion 0.6388707757792675, resistance 0.27214269892338017, influence 1.2244210500557091, and revolt threshold 0.8232588217558955
    Hi, I am Individual 226 with opinion -0.2060926692021674, resistance 0.35087523114180863, influence 1.0050571521589076, and revolt threshold 0.9796503501224724
    Hi, I am Individual 103 with opinion -0.8660157923737679, resistance 0.09749167962163863, influence 1.1694280816121028, and revolt threshold 0.9680943190449652
    Hi, I am Individual 396 with opinion 0.619502474887238, resistance 0.5596625374169203, influence 1.1929809201686858, and revolt threshold 0.9034102148527151
    Hi, I am Individual 998 with opinion -0.8927470825316304, resistance 0.9587143284989594, influence 1.1061730840620552, and revolt threshold 0.7273069102515942
    Hi, I am Individual 596 with opinion 0.32262638022603585, resistance 0.4878631709568044, influence 0.8091183932191267, and revolt threshold 0.7212294815937795
    Hi, I am Individual 46 with opinion 0.5554660417161115, resistance 0.6796239996324139, influence 0.9237694031537433, and revolt threshold 0.9871962715874203
    Hi, I am Individual 562 with opinion -0.9672669428211134, resistance 0.7675120122331445, influence 0.8929170767661307, and revolt threshold 0.9455999022368423
    Hi, I am Individual 7 with opinion 0.5680732331299518, resistance 0.1631615074525834, influence 1.4068325525260654, and revolt threshold 0.8053469179689873
    Hi, I am an propaganda agent, you can call me Propaganda Hub 1049 and I have the id of 1049.       I have the opinion of -1, and influence strength is 0.757815741807265      with a radius of 47.423408813574646.
    Hi, I am Individual 110 with opinion 0.8971892172597884, resistance 0.7634619366060442, influence 1.3484977708964112, and revolt threshold 0.7362274113382912
    Hi, I am Individual 498 with opinion -0.8514930185859886, resistance 0.3776161894869411, influence 0.9781739420117614, and revolt threshold 0.791683284712075
    Hi, I am Individual 393 with opinion -0.767264284556539, resistance 0.8236589973370556, influence 1.4319010355197552, and revolt threshold 0.866738994740672
    Hi, I am Individual 368 with opinion 0.7597669070809696, resistance 0.3261600302531297, influence 1.4445747617723903, and revolt threshold 0.7739933847194749
    Hi, I am Individual 67 with opinion -0.5621246835032252, resistance 0.4674606364706828, influence 1.3059878190440641, and revolt threshold 0.7993840265962389
    Hi, I am Individual 574 with opinion -0.7199141540359384, resistance 0.4530298271051403, influence 0.8465738481852517, and revolt threshold 0.7470679423911968
    Hi, I am Individual 95 with opinion 0.5469181378723511, resistance 0.8853498890825253, influence 1.1605718343787101, and revolt threshold 0.8256247959085485
    Hi, I am Individual 618 with opinion 0.14189328886736896, resistance 0.7263866541914287, influence 0.8925478803197101, and revolt threshold 0.9844423025731364
    Hi, I am Individual 679 with opinion -0.6956267114901402, resistance 0.874520928319801, influence 1.2289225191018245, and revolt threshold 0.8485269695430117
    Hi, I am Individual 290 with opinion -0.8910177814517308, resistance 0.2169787238319456, influence 0.9256576499460087, and revolt threshold 0.9894503481479664
    Hi, I am Individual 70 with opinion -0.4946715568955793, resistance 0.8912025428647394, influence 1.4175351342236016, and revolt threshold 0.8200940093515741
    Hi, I am Individual 777 with opinion 0.3467516008509295, resistance 0.25239085328633193, influence 0.8811447238035824, and revolt threshold 0.9102498472597196
    Hi, I am Individual 375 with opinion 0.8502516158075861, resistance 0.33448666661472193, influence 1.4124791367034144, and revolt threshold 0.8457269842205641
    Hi, I am Individual 676 with opinion -0.5557020425366705, resistance 0.0006141444009076791, influence 1.395742063213298, and revolt threshold 0.7302129924450537
    Hi, I am Individual 439 with opinion -0.4672979125112142, resistance 0.8553174510753383, influence 1.3811741815033745, and revolt threshold 0.7572305805148117
    Hi, I am Individual 355 with opinion 0.06072497499841489, resistance 0.47791575232308114, influence 1.0587580796956355, and revolt threshold 0.976000565799727
    Hi, I am Individual 189 with opinion 0.10664663196302127, resistance 0.9955206532078628, influence 0.8812072435246645, and revolt threshold 0.8857797249292727
    Hi, I am Individual 709 with opinion -0.08287591091191326, resistance 0.411934627170784, influence 1.3634340311465194, and revolt threshold 0.9111854497090257
    Hi, I am Individual 724 with opinion 0.04257113691870784, resistance 0.006558806381346982, influence 0.835024189517863, and revolt threshold 0.9373359908015887
    Hi, I am Individual 80 with opinion 0.7028065009813818, resistance 0.2898991780966189, influence 1.3685888254083856, and revolt threshold 0.7037648234769022
    Hi, I am Individual 126 with opinion -0.05087333370589153, resistance 0.8719163469530583, influence 1.1247950019341202, and revolt threshold 0.9287885315915472
    Hi, I am Individual 471 with opinion 0.1705587938826043, resistance 0.5088948367468298, influence 1.3189803889236464, and revolt threshold 0.9720283970065654
    Hi, I am Individual 9 with opinion 0.4589657837730776, resistance 0.09709088097030871, influence 1.0116089864199156, and revolt threshold 0.9416278907839191
    Hi, I am Individual 828 with opinion -0.7275073274209836, resistance 0.3526237793104501, influence 1.3882524579149307, and revolt threshold 0.9122485425834301
    Hi, I am Individual 631 with opinion 0.9693554663983182, resistance 0.4096846636418936, influence 0.9773380916135384, and revolt threshold 0.8935156702605245
    Hi, I am Individual 568 with opinion -0.4605150479610294, resistance 0.41326784541293937, influence 0.8750598669227881, and revolt threshold 0.9355811943277397
    Hi, I am Individual 425 with opinion 0.02293640754703352, resistance 0.6807243428296594, influence 0.8970071300384923, and revolt threshold 0.955346067843093
    Hi, I am Individual 432 with opinion 0.8259218009409071, resistance 0.5518179694114029, influence 1.2502153004651206, and revolt threshold 0.8691477213172878
    Hi, I am Individual 970 with opinion 0.6411143372522972, resistance 0.5907628160127046, influence 1.2062480559165105, and revolt threshold 0.7582252754564355
    Hi, I am Individual 619 with opinion -0.14962666382919876, resistance 0.3175450102479038, influence 1.3264082833470925, and revolt threshold 0.9139453165903338
    Hi, I am Individual 956 with opinion 0.0660327422217355, resistance 0.7952549370211083, influence 0.9319105819090998, and revolt threshold 0.783109455287256
    Hi, I am Individual 545 with opinion 0.17391488541075906, resistance 0.8233076940152575, influence 1.3779188012545587, and revolt threshold 0.924123526192407
    Hi, I am Individual 144 with opinion 0.4907614323766838, resistance 0.2147881172460807, influence 1.341590485758232, and revolt threshold 0.7762776061930909
    Hi, I am Individual 169 with opinion -0.8470723776110718, resistance 0.49987278608129715, influence 1.1065400050559975, and revolt threshold 0.9463227142875414
    Hi, I am Individual 999 with opinion 0.9477264185206009, resistance 0.4614821996992514, influence 1.3648944169000732, and revolt threshold 0.8826774325878683
    Hi, I am Individual 230 with opinion 0.658364896966519, resistance 0.2417835017308979, influence 0.9393699905775522, and revolt threshold 0.7065147000505388
    Hi, I am Individual 130 with opinion -0.19173020920462802, resistance 0.5211818116260207, influence 1.2578248408259634, and revolt threshold 0.7701943925445869
    Hi, I am Individual 653 with opinion 0.9217481465548787, resistance 0.16699746182259267, influence 1.0870271366548077, and revolt threshold 0.7543874823779281
    Hi, I am Individual 299 with opinion 0.6370588539363826, resistance 0.98038312740175, influence 1.311943295984216, and revolt threshold 0.8256432010065005
    Hi, I am Individual 514 with opinion -0.5087551024668471, resistance 0.22447999312690703, influence 1.3877157221583094, and revolt threshold 0.8990247833950034
    Hi, I am Individual 188 with opinion -0.09225911355915994, resistance 0.5025021504366282, influence 1.416373592381765, and revolt threshold 0.9184052454315746
    Hi, I am Individual 108 with opinion 0.8306873512644011, resistance 0.6983822773063945, influence 1.2426550301810808, and revolt threshold 0.944945755670622
    Hi, I am Individual 751 with opinion 0.5582738926727546, resistance 0.004572249312985832, influence 0.8882358581866797, and revolt threshold 0.7416389801641182
    Hi, I am Individual 669 with opinion -0.1261691676617056, resistance 0.06480995318523675, influence 1.0737956456195135, and revolt threshold 0.9620512811728555
    Hi, I am Individual 667 with opinion 0.2724853704226835, resistance 0.06905253288761914, influence 1.4303690753782348, and revolt threshold 0.9800032149573389
    Hi, I am Individual 308 with opinion -0.7103233498749832, resistance 0.8834999664059064, influence 1.3413584697836443, and revolt threshold 0.9525330507117442
    Hi, I am Individual 813 with opinion -0.043726738372867, resistance 0.915289598357952, influence 1.3338945529551385, and revolt threshold 0.8554778373487312
    Hi, I am Individual 340 with opinion -0.0444437470485981, resistance 0.5079630496623528, influence 0.8611907362620117, and revolt threshold 0.9779616019343365
    Hi, I am an propaganda agent, you can call me Propaganda Hub 1007 and I have the id of 1007.       I have the opinion of 1, and influence strength is 0.7663864590645165      with a radius of 95.75732616720308.
    Hi, I am Individual 646 with opinion -0.7807083939888921, resistance 0.65861887708418, influence 1.2196139974148916, and revolt threshold 0.9280602829423089
    Hi, I am Individual 938 with opinion 0.28211650875341454, resistance 0.5295040011166118, influence 1.1933556725821872, and revolt threshold 0.7551966665147205
    Hi, I am Individual 549 with opinion 0.17699443789193836, resistance 0.8927780765419276, influence 1.0225090461485955, and revolt threshold 0.7768419040494098
    Hi, I am Individual 665 with opinion -0.49282842607213584, resistance 0.33828887303268484, influence 1.4689649810038596, and revolt threshold 0.7490853764413781
    Hi, I am Individual 199 with opinion 0.7424215649963446, resistance 0.21904593723650778, influence 1.151890455966973, and revolt threshold 0.8653475523045286
    Hi, I am Individual 790 with opinion 0.39239924598954956, resistance 0.43522028028167103, influence 1.080240051841854, and revolt threshold 0.9257511274997373
    Hi, I am Individual 351 with opinion 0.110228472353741, resistance 0.9122276548808433, influence 0.8638550998264899, and revolt threshold 0.9256985719287947
    Hi, I am Individual 244 with opinion -0.7982210212016856, resistance 0.6141433788359614, influence 1.401754000698629, and revolt threshold 0.88254581712758
    Hi, I am Individual 807 with opinion 0.2073297826148155, resistance 0.38832475791436905, influence 1.1917981537574278, and revolt threshold 0.9595862861251552
    Hi, I am Individual 627 with opinion -0.021842143601835007, resistance 0.4239914005663057, influence 1.4353930479386041, and revolt threshold 0.7782312395299796
    Hi, I am Individual 395 with opinion -0.45420228499813864, resistance 0.19420574429631532, influence 0.8393191737341874, and revolt threshold 0.8951375957569419
    Hi, I am Individual 344 with opinion -0.6051786816361211, resistance 0.3091553408523541, influence 1.416460098451893, and revolt threshold 0.9979531523214303
    Hi, I am Individual 601 with opinion -0.009444712731935034, resistance 0.655045922736328, influence 1.418132612370663, and revolt threshold 0.8032838028954514
    Hi, I am Individual 985 with opinion -0.5992523716583469, resistance 0.07635219866450116, influence 1.1413724970296226, and revolt threshold 0.828702138675742
    Hi, I am Individual 629 with opinion -0.1613653740207055, resistance 0.24547877871067558, influence 1.2595385748227321, and revolt threshold 0.9411948949740342
    Hi, I am Individual 656 with opinion -0.12039155209082675, resistance 0.547556552934436, influence 1.1073656370441411, and revolt threshold 0.8566618133117253
    Hi, I am Individual 119 with opinion -0.8987975880773968, resistance 0.972662679900574, influence 1.1550207989630867, and revolt threshold 0.9116149413587403
    Hi, I am Individual 17 with opinion -0.7239998373610146, resistance 0.573222745644959, influence 1.350052107322788, and revolt threshold 0.7472122329649435
    Hi, I am Individual 407 with opinion 0.9251167768895929, resistance 0.4094148114444668, influence 1.4639854903655949, and revolt threshold 0.7425189242133168
    Hi, I am Individual 427 with opinion -0.7900003474523285, resistance 0.4109824540171314, influence 0.876306227968767, and revolt threshold 0.8766921935443667
    Hi, I am Individual 609 with opinion -0.29845941226768224, resistance 0.21372093006683668, influence 1.0267258462755622, and revolt threshold 0.7400579377826236
    Hi, I am Individual 394 with opinion -0.7518055745736874, resistance 0.7999535350703593, influence 1.1225305284124736, and revolt threshold 0.8979774329771097
    Hi, I am Individual 532 with opinion 0.5094286452509158, resistance 0.9711698162062231, influence 0.9275745099690643, and revolt threshold 0.9861294072766442
    Hi, I am Individual 991 with opinion -0.28576554056996284, resistance 0.6814507698399195, influence 1.073849921309227, and revolt threshold 0.9111274574222344
    Hi, I am Individual 135 with opinion -0.04173602970659873, resistance 0.5108945059284798, influence 1.1982652745959053, and revolt threshold 0.7343938325372741
    Hi, I am Individual 181 with opinion -0.7202664147579041, resistance 0.3671543069205925, influence 1.478806888364402, and revolt threshold 0.7022124830895942
    Hi, I am Individual 835 with opinion 0.45399609472007096, resistance 0.15440558452405106, influence 0.8374192021636025, and revolt threshold 0.9353172968949474
    Hi, I am Individual 129 with opinion 0.03922071335286881, resistance 0.014617609879795879, influence 0.8426788833659894, and revolt threshold 0.8713344596559496
    Hi, I am Individual 410 with opinion -0.7085245166348344, resistance 0.7218778669337811, influence 0.9168127654363064, and revolt threshold 0.895595540562245
    Hi, I am Individual 968 with opinion 0.9053224639232431, resistance 0.01363412461502822, influence 1.041245643186014, and revolt threshold 0.9995359451939564
    Hi, I am Individual 830 with opinion -0.434242720673516, resistance 0.2809528743179067, influence 0.930741803259096, and revolt threshold 0.8245073586381462
    Hi, I am Individual 493 with opinion 0.13542386178355814, resistance 0.925826995935155, influence 1.0829052070675491, and revolt threshold 0.8173514130486169
    Hi, I am Individual 977 with opinion 0.6252682010346937, resistance 0.9039141360036212, influence 1.3019781892453457, and revolt threshold 0.7103187249439246
    Hi, I am Individual 120 with opinion 0.1556888333634534, resistance 0.1505900440939637, influence 1.0312687850370472, and revolt threshold 0.9506559276349971
    Hi, I am an propaganda agent, you can call me Propaganda Hub 1001 and I have the id of 1001.       I have the opinion of 1, and influence strength is 0.5369653318497886      with a radius of 26.134052804099664.
    Hi, I am Individual 863 with opinion 0.11077937958800588, resistance 0.6011877204338615, influence 1.0322101122700886, and revolt threshold 0.967352283459
    Hi, I am Individual 143 with opinion 0.9387855467338337, resistance 0.20139376007338383, influence 1.1772554972060822, and revolt threshold 0.781757768886949
    Hi, I am Individual 293 with opinion -0.21690122128438927, resistance 0.2224891733242763, influence 1.3630898412592343, and revolt threshold 0.948951477424007
    Hi, I am Individual 452 with opinion -0.8708158188182893, resistance 0.844166375367792, influence 1.2228594708605653, and revolt threshold 0.8416944239817497
    Hi, I am Individual 0 with opinion 0.9472462936186177, resistance 0.32989506008121183, influence 1.0920985376273586, and revolt threshold 0.7528718989754276
    Hi, I am Individual 320 with opinion 0.5711454149531696, resistance 0.29454268915666615, influence 1.1761400953107182, and revolt threshold 0.8735923756147027
    Hi, I am Individual 529 with opinion 0.503648151597107, resistance 0.01627669074123661, influence 1.3217077984044199, and revolt threshold 0.7386773970050995
    Hi, I am Individual 353 with opinion 0.9681799127689155, resistance 0.7015758470989507, influence 1.0306456551525007, and revolt threshold 0.8737391355236879
    Hi, I am Individual 994 with opinion 0.9412647347556649, resistance 0.9909934767937074, influence 1.1786483896708186, and revolt threshold 0.9656330284208707
    Hi, I am Individual 846 with opinion -0.5525377123699371, resistance 0.5508012929515783, influence 1.0868876094232252, and revolt threshold 0.7270811756675432
    Hi, I am an propaganda agent, you can call me Propaganda Hub 1039 and I have the id of 1039.       I have the opinion of -1, and influence strength is 0.6015743729062654      with a radius of 78.6703449342434.
    Hi, I am Individual 106 with opinion -0.6531135194852569, resistance 0.2196478059209367, influence 0.932275077322213, and revolt threshold 0.9688527855813707
    Hi, I am Individual 287 with opinion -0.29855059373723103, resistance 0.026981035729068736, influence 1.0066929897238015, and revolt threshold 0.7736232390834056
    Hi, I am Individual 952 with opinion 0.7826060531603782, resistance 0.928884988524544, influence 1.3451084615685305, and revolt threshold 0.8931065637711681
    Hi, I am Individual 139 with opinion -0.05403459407296207, resistance 0.4931317682044529, influence 1.3624543908668416, and revolt threshold 0.9428860295389935
    Hi, I am Individual 504 with opinion 0.26768119296668935, resistance 0.8014874977572061, influence 0.8837509009869766, and revolt threshold 0.7899796932030327
    Hi, I am Individual 848 with opinion -0.18654135679159345, resistance 0.5750313889729819, influence 1.3969939626617514, and revolt threshold 0.7448314475425029
    Hi, I am Individual 157 with opinion 0.25111127324972204, resistance 0.7953372450261413, influence 1.1373028091850064, and revolt threshold 0.7245978242645407
    Hi, I am Individual 513 with opinion 0.6818147644322814, resistance 0.9749256516024714, influence 0.9234130835555038, and revolt threshold 0.8531606417694836
    Hi, I am Individual 791 with opinion 0.8684444243282086, resistance 0.28842857512027464, influence 1.1615130365712811, and revolt threshold 0.7412807402548446
    Hi, I am Individual 207 with opinion 0.5254414342555374, resistance 0.7934020844149968, influence 1.0930992237371888, and revolt threshold 0.9473559873040883
    Hi, I am Individual 399 with opinion -0.5636824150070123, resistance 0.03544679634769121, influence 0.9269694927112252, and revolt threshold 0.9785468392418083
    Hi, I am Individual 961 with opinion -0.7638281160840743, resistance 0.3115558552948844, influence 0.9468017149010032, and revolt threshold 0.764029745156266
    Hi, I am Individual 392 with opinion -0.8415503012908236, resistance 0.5268923443543376, influence 1.4249686284101175, and revolt threshold 0.9330555177651174
    Hi, I am Individual 644 with opinion 0.5560749209303617, resistance 0.5858973864734938, influence 1.1573915926630178, and revolt threshold 0.8335682297515956
    Hi, I am Individual 741 with opinion 0.19699621870259887, resistance 0.7730394449212262, influence 1.1293376196313132, and revolt threshold 0.8366892399840137
    Hi, I am Individual 508 with opinion -0.5804264545463909, resistance 0.13982019190643336, influence 1.4038182620992212, and revolt threshold 0.7046235469624411
    Hi, I am Individual 988 with opinion -0.5188784481735078, resistance 0.19624802078205772, influence 0.8413151852033474, and revolt threshold 0.988255906959862
    Hi, I am Individual 348 with opinion 0.12853845951345155, resistance 0.8144108173505811, influence 1.068941580563131, and revolt threshold 0.7783505606191021
    Hi, I am Individual 104 with opinion -0.502808753843567, resistance 0.3901614536112451, influence 0.8093240472395584, and revolt threshold 0.742271286115496
    Hi, I am Individual 78 with opinion -0.6132571952850363, resistance 0.17917947262905765, influence 1.1750285075639546, and revolt threshold 0.8920315208264653
    Hi, I am Individual 478 with opinion -0.9478676965303166, resistance 0.43888829734484225, influence 0.8683097352272855, and revolt threshold 0.7099707726162119
    Hi, I am Individual 847 with opinion -0.13340513304610235, resistance 0.13949141655715247, influence 1.1848796384813718, and revolt threshold 0.7510592917807194
    Hi, I am Individual 766 with opinion -0.8175389868233178, resistance 0.660791368803638, influence 1.3827690998320572, and revolt threshold 0.891656259852053
    Hi, I am Individual 214 with opinion 0.05788560406292875, resistance 0.8773096594603091, influence 0.9570649135149738, and revolt threshold 0.8995956962493135
    Hi, I am Individual 428 with opinion 0.470091247494391, resistance 0.38762190463580326, influence 0.9605313975309857, and revolt threshold 0.9142436669648847
    Hi, I am Individual 40 with opinion -0.33202510300430865, resistance 0.39131586564172904, influence 1.183904489588885, and revolt threshold 0.8460406470704911
    Hi, I am Individual 931 with opinion 0.6741725189993029, resistance 0.4489995046259395, influence 1.0476844088354127, and revolt threshold 0.7598903065337332
    Hi, I am Individual 136 with opinion -0.3901663116294045, resistance 0.6723061292060789, influence 1.3394296763284335, and revolt threshold 0.7898308891530239
    Hi, I am Individual 971 with opinion -0.6541848184986849, resistance 0.4766178600109895, influence 1.4410927700195904, and revolt threshold 0.9776422640249232
    Hi, I am Individual 550 with opinion -0.4371262991712992, resistance 0.3898374180593903, influence 0.9206861269474171, and revolt threshold 0.8879951518259341
    Hi, I am Individual 182 with opinion 0.4148613668376657, resistance 0.5361547237679065, influence 1.3036835930995578, and revolt threshold 0.9862102144094211
    Hi, I am Individual 882 with opinion 0.09760151302528186, resistance 0.7765880746168413, influence 0.9004912332867783, and revolt threshold 0.8553021541710293
    Hi, I am Individual 888 with opinion 0.6738884441572444, resistance 0.9217792774638582, influence 1.334867535263803, and revolt threshold 0.7035497067050518
    Hi, I am Individual 712 with opinion 0.37520864999852543, resistance 0.6392774521947476, influence 1.257871328335665, and revolt threshold 0.7344685295387963
    Hi, I am Individual 499 with opinion -0.9288282428206884, resistance 0.3552438216263125, influence 1.4428472298315753, and revolt threshold 0.8840783545521972
    Hi, I am Individual 909 with opinion 0.027826597783432883, resistance 0.21289184004399553, influence 0.8475502735337748, and revolt threshold 0.8183407670846032
    Hi, I am Individual 862 with opinion -0.8884451389568198, resistance 0.42126700569424913, influence 1.3496495621442897, and revolt threshold 0.8003638528926915
    Hi, I am Individual 519 with opinion 0.9901683253881299, resistance 0.3576804714829459, influence 0.880726433702979, and revolt threshold 0.9268852963138932
    Hi, I am Individual 671 with opinion 0.49971363523582535, resistance 0.9147630031085264, influence 0.9638596485282098, and revolt threshold 0.8988771623837608
    Hi, I am Individual 133 with opinion 0.5446143803129166, resistance 0.8473572333175706, influence 1.4534652623260238, and revolt threshold 0.826559138891642
    Hi, I am Individual 856 with opinion -0.631778488762663, resistance 0.29941932034297947, influence 0.9233121115311672, and revolt threshold 0.7092968529282503
    Hi, I am Individual 292 with opinion 0.6119595541919942, resistance 0.059538599516182567, influence 1.0212245586041162, and revolt threshold 0.8164536804952355
    Hi, I am Individual 767 with opinion 0.9093187380361898, resistance 0.7129214092943839, influence 0.8227090453091926, and revolt threshold 0.7053988898576169
    Hi, I am Individual 965 with opinion -0.5011430089727165, resistance 0.2521771222503265, influence 1.186099444903077, and revolt threshold 0.8103697015839905
    Hi, I am Individual 844 with opinion -0.7715810625620905, resistance 0.6319869343714202, influence 1.4929098656335729, and revolt threshold 0.7121847232870284
    Hi, I am Individual 420 with opinion -0.11338043944527842, resistance 0.094053303199369, influence 1.2048135048555388, and revolt threshold 0.8745035859376058
    Hi, I am Individual 220 with opinion -0.6920232482109896, resistance 0.17878108765470024, influence 0.9536236599305433, and revolt threshold 0.8449570293266353
    Hi, I am Individual 745 with opinion -0.65462702522212, resistance 0.7347064123955719, influence 0.9608345322325261, and revolt threshold 0.7689132945523737
    Hi, I am Individual 870 with opinion 0.07984271436083046, resistance 0.5422385596106241, influence 1.0149647475448675, and revolt threshold 0.719528065664495
    Hi, I am Individual 748 with opinion -0.5634280226155484, resistance 0.41123727192947934, influence 1.1978190361360022, and revolt threshold 0.7764938907758497
    Hi, I am Individual 664 with opinion 0.5202951410434986, resistance 0.3043893565659197, influence 1.0366203291027491, and revolt threshold 0.9750256342079036
    Hi, I am Individual 172 with opinion -0.3526105697403823, resistance 0.3618556665464897, influence 0.937744693371907, and revolt threshold 0.9532343268162091
    Hi, I am Individual 757 with opinion -0.8433993806669893, resistance 0.23101220487276553, influence 1.2932208376092762, and revolt threshold 0.7276695242421751
    Hi, I am Individual 423 with opinion 0.7055646179148709, resistance 0.7845714183746553, influence 1.1816763246174833, and revolt threshold 0.8410528150512848
    Hi, I am Individual 808 with opinion -0.5959391493670756, resistance 0.832377570773647, influence 1.2871662908840888, and revolt threshold 0.8610438062782305
    Hi, I am Individual 517 with opinion -0.6276282494690442, resistance 0.44670602539138904, influence 0.994857002787652, and revolt threshold 0.9152480870167161
    Hi, I am Individual 675 with opinion 0.4027584906853212, resistance 0.8943754137490351, influence 1.119561617225201, and revolt threshold 0.749334816778651
    Hi, I am Individual 580 with opinion 0.2323847719026677, resistance 0.22651176335592393, influence 1.256594322334222, and revolt threshold 0.8849723166091035
    Hi, I am Individual 875 with opinion -0.6951104874047933, resistance 0.5813493288163117, influence 0.9034321470499365, and revolt threshold 0.885674894732845
    Hi, I am Individual 869 with opinion 0.5481779489666947, resistance 0.023311634121927827, influence 0.9901243836911067, and revolt threshold 0.7632847829566765
    Hi, I am Individual 704 with opinion 0.24287171605896263, resistance 0.5982745448397092, influence 1.3678390751207707, and revolt threshold 0.7461746193744975
    Hi, I am Individual 447 with opinion 0.5286972761301627, resistance 0.4848602566630904, influence 1.0290645085278816, and revolt threshold 0.8452199734078493
    Hi, I am Individual 190 with opinion -0.39577575398919773, resistance 0.3050587869040824, influence 1.117389507579325, and revolt threshold 0.85839240739164
    Hi, I am Individual 834 with opinion -0.02937612944457446, resistance 0.04200814687902921, influence 1.3643911318124564, and revolt threshold 0.8285808319293212
    Hi, I am Individual 904 with opinion -0.8966884928618679, resistance 0.9487058900819427, influence 1.010384940143254, and revolt threshold 0.9443991459325468
    Hi, I am Individual 613 with opinion -0.1967582718131664, resistance 0.6517513320734608, influence 1.3520060076962421, and revolt threshold 0.943246860347001
    Hi, I am Individual 337 with opinion -0.0631897857236281, resistance 0.3170095004276948, influence 1.308393914069498, and revolt threshold 0.8888531584784815
    Hi, I am Individual 683 with opinion 0.6969369719768048, resistance 0.12103253413162929, influence 1.1646526703375462, and revolt threshold 0.8275278383834711
    Hi, I am Individual 8 with opinion -0.5885672392592012, resistance 0.3352543564191437, influence 1.3360392579470122, and revolt threshold 0.8372514049000483
    Hi, I am Individual 202 with opinion -0.8102622116206097, resistance 0.5166645712838516, influence 1.2322843675375292, and revolt threshold 0.740635629736939
    Hi, I am Individual 850 with opinion -0.6597855675900428, resistance 0.6312787652644524, influence 1.283349370153196, and revolt threshold 0.8701916202881818
    Hi, I am Individual 161 with opinion 0.9558740135617834, resistance 0.43328877123289156, influence 1.2159885142445161, and revolt threshold 0.7357371264294916
    Hi, I am Individual 284 with opinion 0.7036434437624786, resistance 0.28627894528761655, influence 1.4600535220634474, and revolt threshold 0.9849076842856881
    Hi, I am Individual 623 with opinion -0.9015310918819885, resistance 0.6651458161834956, influence 1.2151462567463227, and revolt threshold 0.8398968488074366
    Hi, I am Individual 315 with opinion 0.16815520827540742, resistance 0.2766466567740108, influence 1.292125597812773, and revolt threshold 0.996279308017786
    Hi, I am Individual 102 with opinion 0.7614752715655102, resistance 0.8621855129787803, influence 1.1327340428963404, and revolt threshold 0.7069506656771135
    Hi, I am an propaganda agent, you can call me Propaganda Hub 1000 and I have the id of 1000.       I have the opinion of -1, and influence strength is 0.8925840725471785      with a radius of 63.08656586693857.
    Hi, I am Individual 874 with opinion 0.4241073362726364, resistance 0.06778865617925667, influence 1.2128486127749156, and revolt threshold 0.889875768707817
    Hi, I am Individual 192 with opinion -0.8902858493551864, resistance 0.11897752557939101, influence 1.263788183834179, and revolt threshold 0.7841003751921212
    Hi, I am Individual 528 with opinion 0.6242044673139604, resistance 0.008626585420004829, influence 1.1238993760376188, and revolt threshold 0.8627558037102113
    Hi, I am Individual 908 with opinion -0.841388566682234, resistance 0.7570074751848537, influence 1.4866522540612572, and revolt threshold 0.7625400291941911
    Hi, I am Individual 186 with opinion 0.7552993748478822, resistance 0.7380549730069982, influence 1.2501919612260897, and revolt threshold 0.8877675437326379
    Hi, I am Individual 806 with opinion -0.9615530720473258, resistance 0.5417500772795725, influence 1.324996131263597, and revolt threshold 0.9559578850365975
    Hi, I am an propaganda agent, you can call me Propaganda Hub 1037 and I have the id of 1037.       I have the opinion of -1, and influence strength is 0.41032863650305973      with a radius of 82.5157525054401.
    Hi, I am Individual 1 with opinion 0.6596038524603962, resistance 0.8443636250191671, influence 0.880428394412025, and revolt threshold 0.9500533031625973
    Hi, I am Individual 802 with opinion 0.8626629918154256, resistance 0.5326231604079805, influence 0.9358710618594266, and revolt threshold 0.9086048579993273
    Hi, I am Individual 44 with opinion -0.15219851267382833, resistance 0.16278830939618616, influence 0.970265256572574, and revolt threshold 0.7135461327468843
    Hi, I am Individual 10 with opinion 0.9627644746002435, resistance 0.391388032478415, influence 1.425631228010813, and revolt threshold 0.9963061798205937
    Hi, I am an propaganda agent, you can call me Propaganda Hub 1030 and I have the id of 1030.       I have the opinion of 1, and influence strength is 0.8594915433672011      with a radius of 28.226969359059268.
    Hi, I am Individual 124 with opinion -0.8651399285394208, resistance 0.10204430462192304, influence 1.2524580489505375, and revolt threshold 0.7171048080413028
    Hi, I am Individual 457 with opinion 0.6452083603061329, resistance 0.38759456609906384, influence 1.0545813411212008, and revolt threshold 0.8681780838905714
    Hi, I am Individual 317 with opinion 0.27654982489177704, resistance 0.8144796996526966, influence 0.8620817678512522, and revolt threshold 0.9503156792914009
    Hi, I am Individual 233 with opinion -0.5865896274940248, resistance 0.2218255211715212, influence 1.1045227324419875, and revolt threshold 0.7282348639651329
    Hi, I am Individual 868 with opinion -0.45897889885187504, resistance 0.8652597698500352, influence 1.079321253041237, and revolt threshold 0.7112051730915377
    Hi, I am Individual 768 with opinion 0.11163130925401155, resistance 0.25087598131924127, influence 1.0985792108849202, and revolt threshold 0.84707047486003
    Hi, I am Individual 221 with opinion -0.37382003176699796, resistance 0.6542116370422484, influence 0.8304100885389833, and revolt threshold 0.7942371788877618
    Hi, I am Individual 891 with opinion 0.14415297885313283, resistance 0.9346165143547456, influence 1.4127016934767043, and revolt threshold 0.9496858856102999
    Hi, I am Individual 542 with opinion 0.41230631241991667, resistance 0.22946331922459828, influence 1.280682540899761, and revolt threshold 0.9240912874594844
    Hi, I am Individual 895 with opinion 0.6852283601872029, resistance 0.3314835486964176, influence 1.2391885876324498, and revolt threshold 0.7594634978646272
    Hi, I am Individual 373 with opinion 0.06481996721117222, resistance 0.21486329753823186, influence 1.3589569657178506, and revolt threshold 0.8285904893203659
    Hi, I am Individual 342 with opinion 0.2985254703160274, resistance 0.8740856417437983, influence 0.9601296250090856, and revolt threshold 0.77516838083652
    Hi, I am Individual 43 with opinion 0.5526137703120293, resistance 0.37773262356321835, influence 1.2351172417119387, and revolt threshold 0.8164185670610714
    Hi, I am an propaganda agent, you can call me Propaganda Hub 1005 and I have the id of 1005.       I have the opinion of -1, and influence strength is 0.9832016527620692      with a radius of 53.04435638574593.
    Hi, I am Individual 176 with opinion -0.7561872802495846, resistance 0.26236884982861186, influence 1.1377526539396565, and revolt threshold 0.9015180570585677
    Hi, I am Individual 959 with opinion 0.07613515506359003, resistance 0.22415474777260225, influence 1.4537940902766195, and revolt threshold 0.8978762701647613
    Hi, I am Individual 203 with opinion -0.8544373115721113, resistance 0.027093375286408095, influence 1.0760459783137364, and revolt threshold 0.859047683886949
    Hi, I am Individual 694 with opinion -0.2840341201939627, resistance 0.9694761606207006, influence 0.9228043789593553, and revolt threshold 0.7025440282072322
    Hi, I am Individual 387 with opinion -0.6469389639425223, resistance 0.9988730728563632, influence 1.2350991378262834, and revolt threshold 0.7857575586295196
    Hi, I am Individual 771 with opinion 0.16427833567684735, resistance 0.7531315659777189, influence 1.459404594957946, and revolt threshold 0.7625439713138918
    Hi, I am Individual 571 with opinion 0.6994189934951529, resistance 0.25344708053622933, influence 1.3922756450787235, and revolt threshold 0.8523817034123384
    Hi, I am Individual 788 with opinion -0.9726383965452554, resistance 0.8295691625034023, influence 1.068966360249555, and revolt threshold 0.7098962391021726
    Hi, I am Individual 925 with opinion 0.6622478702779613, resistance 0.1660428691187794, influence 1.1154030156266894, and revolt threshold 0.7885336069320256
    Hi, I am Individual 492 with opinion -0.397064530843972, resistance 0.1490956382689015, influence 1.4372387031004281, and revolt threshold 0.9346960242064373
    Hi, I am Individual 852 with opinion 0.9648651853065615, resistance 0.18948326853799236, influence 0.8658352953734664, and revolt threshold 0.8747762191458972
    Hi, I am Individual 811 with opinion 0.777354580421139, resistance 0.23939197971972825, influence 1.2791688881539953, and revolt threshold 0.7174105914125578
    Hi, I am Individual 173 with opinion 0.30735120839113006, resistance 0.9487622255484994, influence 1.2733716266545512, and revolt threshold 0.9646966341740789
    Hi, I am Individual 464 with opinion -0.7349417869058621, resistance 0.5651893981426583, influence 1.3639555113397985, and revolt threshold 0.7643200906610228
    Hi, I am Individual 397 with opinion 0.07065789429599256, resistance 0.46994317998669166, influence 1.1907354387569424, and revolt threshold 0.8263915131248519
    Hi, I am Individual 28 with opinion 0.24299705820237794, resistance 0.2267600758685996, influence 1.2171897634415758, and revolt threshold 0.8578163001296353
    Hi, I am Individual 537 with opinion 0.8011985015889758, resistance 0.633794265861329, influence 1.2628947649625308, and revolt threshold 0.7361409425258442
    Hi, I am Individual 889 with opinion 0.7074569378590747, resistance 0.43823857757131623, influence 1.08392515967751, and revolt threshold 0.8130382709149045
    Hi, I am Individual 312 with opinion -0.7117322713079628, resistance 0.6349715581516288, influence 0.9154899650068117, and revolt threshold 0.8725335504188496
    Hi, I am Individual 440 with opinion 0.2793011547511559, resistance 0.735825491436413, influence 0.8197053191843819, and revolt threshold 0.7062994668669181
    Hi, I am Individual 783 with opinion -0.36033152022715464, resistance 0.7427620458945001, influence 0.8970747536867891, and revolt threshold 0.792912838591346
    Hi, I am Individual 872 with opinion 0.36583874778397885, resistance 0.8168144260214415, influence 1.039522264414507, and revolt threshold 0.7458001018659902
    Hi, I am Individual 398 with opinion -0.8250168262706692, resistance 0.6181546853082719, influence 1.4910157096111671, and revolt threshold 0.8248173641868273
    Hi, I am Individual 593 with opinion -0.7915935385369042, resistance 0.531577704859943, influence 1.0808006545346918, and revolt threshold 0.7764517320318951
    Hi, I am Individual 458 with opinion 0.8823449203065727, resistance 0.8827742542348671, influence 1.2682400670823795, and revolt threshold 0.7640254586519104
    Hi, I am Individual 541 with opinion 0.8494365782208984, resistance 0.49986167942393134, influence 1.0363746470999946, and revolt threshold 0.84337817258695
    Hi, I am Individual 15 with opinion -0.8937975572091459, resistance 0.9173765550995129, influence 1.3851736339122362, and revolt threshold 0.7666197295748514
    Hi, I am Individual 162 with opinion -0.10420315605768882, resistance 0.782307780770607, influence 1.4981445716637543, and revolt threshold 0.735933092745658
    Hi, I am Individual 770 with opinion -0.07437528025848894, resistance 0.11071511274895729, influence 0.8102315414114997, and revolt threshold 0.9773694662651624
    Hi, I am Individual 105 with opinion 0.5931625417321393, resistance 0.5222161475787709, influence 1.1326735675230484, and revolt threshold 0.9581224235368869
    Hi, I am Individual 328 with opinion 0.9038817986510208, resistance 0.19534582398060418, influence 1.3672045480146637, and revolt threshold 0.7315998397664633
    Hi, I am Individual 238 with opinion 0.893514250527504, resistance 0.876698881278471, influence 1.3975178670804045, and revolt threshold 0.9739771464514841
    Hi, I am Individual 215 with opinion 0.19482385511888367, resistance 0.007519052410350113, influence 1.2807297064455747, and revolt threshold 0.824810012254839
    Hi, I am Individual 981 with opinion 0.7082953390519231, resistance 0.20466198136604785, influence 0.8264346369799875, and revolt threshold 0.9591491448895935
    Hi, I am Individual 842 with opinion -0.023053781944685436, resistance 0.05338596699555598, influence 1.009200017140332, and revolt threshold 0.9061636680634373
    Hi, I am Individual 933 with opinion -0.5940533979931362, resistance 0.4329751471070581, influence 1.455683603722457, and revolt threshold 0.8416207143175458
    Hi, I am Individual 907 with opinion 0.9020630470670159, resistance 0.7848348778169985, influence 1.125749160101138, and revolt threshold 0.9790981544216599
    Hi, I am Individual 127 with opinion -0.2385969295533381, resistance 0.40619789908189174, influence 0.9110449108226981, and revolt threshold 0.8398808900769198
    Hi, I am an propaganda agent, you can call me Propaganda Hub 1021 and I have the id of 1021.       I have the opinion of -1, and influence strength is 0.6241467795665054      with a radius of 72.8776435764776.
    Hi, I am Individual 599 with opinion 0.43207557788749984, resistance 0.9408994304778292, influence 1.390203769317619, and revolt threshold 0.9074599044769205
    Hi, I am Individual 840 with opinion -0.09400673965488959, resistance 0.917025208574399, influence 0.8676582060758585, and revolt threshold 0.8933677148374609
    Hi, I am Individual 572 with opinion -0.42824384159198403, resistance 0.9115215353739576, influence 0.944002854502808, and revolt threshold 0.7872274227925774
    Hi, I am Individual 915 with opinion 0.7979548280572235, resistance 0.7822842442378494, influence 1.4768300038367002, and revolt threshold 0.8512275566219147
    Hi, I am Individual 554 with opinion -0.9635366041917546, resistance 0.2497827251873942, influence 0.9924242129475541, and revolt threshold 0.9138611804503243
    Hi, I am Individual 179 with opinion -0.2874845649856408, resistance 0.7555954398967917, influence 1.2323399889626425, and revolt threshold 0.8177196192122479
    Hi, I am Individual 209 with opinion 0.5192248859095512, resistance 0.20872343337532184, influence 1.380903114234734, and revolt threshold 0.9989876962557257
    Hi, I am Individual 725 with opinion -0.7355151176939043, resistance 0.32212620379478707, influence 1.004134749864383, and revolt threshold 0.7549676016412586
    Hi, I am Individual 821 with opinion -0.5908982067050468, resistance 0.829198024304321, influence 1.1260673754291224, and revolt threshold 0.8150271139076481
    Hi, I am Individual 829 with opinion 0.4884657165751811, resistance 0.5268163808339528, influence 1.1528662329796286, and revolt threshold 0.9012542308088123
    Hi, I am Individual 614 with opinion -0.9318470136468562, resistance 0.146335318774221, influence 0.9023336228793254, and revolt threshold 0.8592789388579144
    Hi, I am an propaganda agent, you can call me Propaganda Hub 1024 and I have the id of 1024.       I have the opinion of 1, and influence strength is 0.9691390133570679      with a radius of 75.47977597771158.
    Hi, I am Individual 243 with opinion 0.674606338130137, resistance 0.25892203583603945, influence 0.9483442441147363, and revolt threshold 0.9746973706904687
    Hi, I am Individual 311 with opinion -0.3079449236986431, resistance 0.35489457194858154, influence 0.8218891176075177, and revolt threshold 0.8899226875163523
    Hi, I am Individual 941 with opinion 0.6278956985892503, resistance 0.48562356924011274, influence 1.4190818227254558, and revolt threshold 0.8574568528389366
    Hi, I am Individual 649 with opinion -0.04310938544760501, resistance 0.3903911865718448, influence 1.0311527291375104, and revolt threshold 0.8579525561278186
    Hi, I am Individual 967 with opinion -0.7294542213348003, resistance 0.8004969247390192, influence 0.8586060511562833, and revolt threshold 0.7641039515030985
    Hi, I am Individual 595 with opinion -0.3471587355674326, resistance 0.8043558758240447, influence 1.4292553009933955, and revolt threshold 0.7899092854214147
    Hi, I am Individual 610 with opinion 0.7664361580165044, resistance 0.636137321121676, influence 1.236884926947078, and revolt threshold 0.8158457163201288
    Hi, I am an propaganda agent, you can call me Propaganda Hub 1029 and I have the id of 1029.       I have the opinion of -1, and influence strength is 0.566122012870026      with a radius of 23.537032398327916.
    Hi, I am Individual 866 with opinion 0.8466572188081027, resistance 0.5995142390177961, influence 1.1552712614945642, and revolt threshold 0.9639427941982373
    Hi, I am Individual 112 with opinion -0.23546724292613952, resistance 0.7699669114912491, influence 1.3740249568237468, and revolt threshold 0.8399410004722025
    Hi, I am Individual 77 with opinion 0.8414739942072942, resistance 0.59567078221897, influence 0.8512207970515029, and revolt threshold 0.8750474170761876
    Hi, I am Individual 780 with opinion 0.46162070452496984, resistance 0.012872119660285897, influence 0.911818740832262, and revolt threshold 0.7870799722043788
    Hi, I am Individual 151 with opinion 0.6020552207199494, resistance 0.4891778610898716, influence 1.1229853739545972, and revolt threshold 0.7647692970568271
    Hi, I am Individual 83 with opinion 0.5305857437925869, resistance 0.9943453714769069, influence 1.1237521252096654, and revolt threshold 0.980297305198259
    Hi, I am Individual 72 with opinion 0.26150832698661564, resistance 0.6024705415068141, influence 1.206281980324421, and revolt threshold 0.9086230988797515
    Hi, I am Individual 253 with opinion -0.28510430091411365, resistance 0.7910216686845828, influence 1.4672049194543484, and revolt threshold 0.9187746198870741
    Hi, I am Individual 252 with opinion 0.19246442575925626, resistance 0.9828501712901463, influence 1.0870543714006375, and revolt threshold 0.7732848852837447
    Hi, I am Individual 242 with opinion -0.5780782436144569, resistance 0.9060343325384382, influence 1.2219857979868625, and revolt threshold 0.7371669737965242
    Hi, I am Individual 951 with opinion 0.8346431330253947, resistance 0.5316446797277182, influence 1.4382348243574887, and revolt threshold 0.747585011458655
    Hi, I am Individual 534 with opinion 0.6201482455018708, resistance 0.3347666942239964, influence 1.206214877886235, and revolt threshold 0.8596419149201572
    Hi, I am Individual 213 with opinion 0.9939805523849279, resistance 0.9101322850214294, influence 0.9146152427339216, and revolt threshold 0.8600233096116182
    Hi, I am Individual 553 with opinion -0.7830923589464338, resistance 0.10136681841163198, influence 1.3578493551809603, and revolt threshold 0.9332337677496054
    Hi, I am Individual 59 with opinion 0.6129666514527627, resistance 0.6385808466378504, influence 1.026177968600731, and revolt threshold 0.925099705288761
    Hi, I am Individual 131 with opinion -0.017049374995821154, resistance 0.8144244135016765, influence 1.126267990072643, and revolt threshold 0.9359785490725541
    Hi, I am Individual 642 with opinion 0.5438207862561413, resistance 0.07388965386452617, influence 1.4836766669505632, and revolt threshold 0.7183874184561457
    Hi, I am an propaganda agent, you can call me Propaganda Hub 1040 and I have the id of 1040.       I have the opinion of 1, and influence strength is 0.9502194019829495      with a radius of 92.05965762854875.
    Hi, I am Individual 240 with opinion -0.4496931167969851, resistance 0.6321731175774327, influence 1.141114773694402, and revolt threshold 0.7074234333859272
    Hi, I am Individual 79 with opinion -0.027558657668625974, resistance 0.5750529415683637, influence 1.052790828993454, and revolt threshold 0.8468445608042808
    Hi, I am Individual 883 with opinion -0.07940517635304856, resistance 0.31741947235391876, influence 1.410178143204445, and revolt threshold 0.722647493731942
    Hi, I am Individual 630 with opinion -0.585954905555991, resistance 0.660177282943651, influence 0.9700926554458844, and revolt threshold 0.808003608745469
    Hi, I am Individual 539 with opinion -0.8565936923674529, resistance 0.7816521827094473, influence 1.3581788966464685, and revolt threshold 0.9962134837529044
    Hi, I am an propaganda agent, you can call me Propaganda Hub 1009 and I have the id of 1009.       I have the opinion of -1, and influence strength is 0.8307293367499193      with a radius of 45.12108792253481.
    Hi, I am Individual 635 with opinion 0.989563784621196, resistance 0.9247967288979143, influence 1.3026317087594284, and revolt threshold 0.8593642297547531
    Hi, I am Individual 697 with opinion 0.8947040199024998, resistance 0.7626870631964205, influence 0.8620040273192038, and revolt threshold 0.9376011892899513
    Hi, I am Individual 502 with opinion -0.3146867530783686, resistance 0.756963689650438, influence 1.282010115620487, and revolt threshold 0.9691045666875249
    Hi, I am Individual 219 with opinion 0.5167206115689698, resistance 0.9263108058383932, influence 1.4199624932330615, and revolt threshold 0.8411262748601539
    Hi, I am Individual 734 with opinion -0.5154042137437176, resistance 0.09131779002449836, influence 0.8810332300668042, and revolt threshold 0.7630804597270447
    Hi, I am Individual 674 with opinion 0.013627992325472293, resistance 0.8105549678512765, influence 1.4290394471193444, and revolt threshold 0.7548263810826636
    Hi, I am Individual 769 with opinion 0.761456239456411, resistance 0.7217245702006383, influence 1.4990749275455173, and revolt threshold 0.935253781330158
    Hi, I am an propaganda agent, you can call me Propaganda Hub 1025 and I have the id of 1025.       I have the opinion of -1, and influence strength is 0.8131915321370224      with a radius of 16.88441980484753.
    Hi, I am Individual 578 with opinion -0.9706679834676908, resistance 0.21759522137720733, influence 0.9474183808040805, and revolt threshold 0.9502793119798791
    Hi, I am Individual 4 with opinion 0.12380691512795194, resistance 0.8243724423773263, influence 0.9680163102679997, and revolt threshold 0.7807237328096296
    Hi, I am Individual 283 with opinion -0.6365485667796402, resistance 0.5667327828183744, influence 1.3410816850056582, and revolt threshold 0.9190149785625483
    Hi, I am an propaganda agent, you can call me Propaganda Hub 1013 and I have the id of 1013.       I have the opinion of 1, and influence strength is 0.44985135040717256      with a radius of 35.17798922634867.
    Hi, I am Individual 66 with opinion -0.3969013714689684, resistance 0.8046865965692263, influence 1.437445029383062, and revolt threshold 0.8616309877072559
    Hi, I am Individual 754 with opinion -0.20805544540000964, resistance 0.9126955777939977, influence 0.8641320315324754, and revolt threshold 0.7139204779972216
    Hi, I am Individual 193 with opinion 0.3977546889204133, resistance 0.023921076772012495, influence 1.022893770319875, and revolt threshold 0.7742081445257675
    Hi, I am Individual 512 with opinion -0.23657162681369148, resistance 0.744754127225008, influence 1.0519023550050626, and revolt threshold 0.722164164624883
    Hi, I am Individual 76 with opinion -0.5040745665048754, resistance 0.8432390041789833, influence 0.855343028988531, and revolt threshold 0.9439289730730756
    Hi, I am Individual 890 with opinion 0.23691435224712176, resistance 0.5951706635205699, influence 1.2844471280024654, and revolt threshold 0.8427138953454151
    Hi, I am Individual 379 with opinion -0.9605832816086779, resistance 0.9100390320108027, influence 1.1671255585436628, and revolt threshold 0.8459821505896813
    Hi, I am Individual 943 with opinion -0.3538397361067742, resistance 0.18911452632983328, influence 1.1728636779633437, and revolt threshold 0.9200786910616201
    Hi, I am an propaganda agent, you can call me Propaganda Hub 1027 and I have the id of 1027.       I have the opinion of 1, and influence strength is 0.9295466588851932      with a radius of 53.81363391148981.
    Hi, I am Individual 648 with opinion -0.42976942067969515, resistance 0.02343894384328138, influence 1.49722700995454, and revolt threshold 0.7291375294539415
    Hi, I am Individual 666 with opinion 0.3337755399467701, resistance 0.738898588949059, influence 1.2296228104012865, and revolt threshold 0.8160660274848527
    Hi, I am Individual 326 with opinion -0.9967361306002838, resistance 0.5388277450846586, influence 1.3139681565192727, and revolt threshold 0.7742461202563018
    Hi, I am Individual 89 with opinion 0.12427460014657599, resistance 0.713832270035576, influence 1.4371086226094243, and revolt threshold 0.7660465376234558
    Hi, I am Individual 974 with opinion 0.812300977728792, resistance 0.7650881286408653, influence 1.2363728562827438, and revolt threshold 0.9338451611406489
    Hi, I am Individual 708 with opinion 0.2754778499701138, resistance 0.20514442209936035, influence 1.3221227840314111, and revolt threshold 0.7941355068525107
    Hi, I am an propaganda agent, you can call me Propaganda Hub 1034 and I have the id of 1034.       I have the opinion of 1, and influence strength is 0.5834263619724331      with a radius of 91.2563095190987.
    Hi, I am Individual 711 with opinion 0.9675753555771176, resistance 0.18285171953200963, influence 1.3452036337419249, and revolt threshold 0.8935774785772779
    Hi, I am Individual 88 with opinion -0.35991078816070643, resistance 0.5587202929464773, influence 0.8279993908179515, and revolt threshold 0.8701076356671135
    Hi, I am Individual 210 with opinion -0.4097047783492893, resistance 0.20633824297814485, influence 1.0412397523126096, and revolt threshold 0.9904771070041267
    Hi, I am Individual 154 with opinion 0.47257605001718694, resistance 0.027314557392824068, influence 1.2046168658934953, and revolt threshold 0.9913734511094787
    Hi, I am Individual 966 with opinion -0.05301485313641918, resistance 0.08046936842301533, influence 0.8307961913459575, and revolt threshold 0.7841573705539578
    Hi, I am Individual 699 with opinion 0.26703759393589976, resistance 0.13032166932135747, influence 1.053803045277185, and revolt threshold 0.7747302463193019
    Hi, I am Individual 68 with opinion 0.3104722892995533, resistance 0.638276689261779, influence 0.9272061711740469, and revolt threshold 0.7346849310482548
    Hi, I am Individual 485 with opinion -0.7857610177005341, resistance 0.030398962472823032, influence 1.2893769237546937, and revolt threshold 0.7761888635801939
    Hi, I am Individual 187 with opinion -0.17437354269216665, resistance 0.1766431486448582, influence 1.4714573813241976, and revolt threshold 0.7226095971717139
    Hi, I am Individual 149 with opinion 0.8408829558159956, resistance 0.21002513267441958, influence 1.2846476888407317, and revolt threshold 0.841668838033968
    Hi, I am Individual 540 with opinion -0.366126402859843, resistance 0.5133424109153161, influence 1.0262176599764337, and revolt threshold 0.931751801036628
    Hi, I am Individual 902 with opinion -0.8929434039894231, resistance 0.513134770516117, influence 1.011290702574885, and revolt threshold 0.9008109739636356
    Hi, I am Individual 268 with opinion 0.4540877586627756, resistance 0.7934823133741685, influence 1.349006674371061, and revolt threshold 0.9565920873370475
    Hi, I am Individual 309 with opinion 0.5091968640339952, resistance 0.39000884104575695, influence 0.827956908782178, and revolt threshold 0.7378192458511778
    Hi, I am Individual 271 with opinion 0.01600944990040354, resistance 0.40464292759906617, influence 0.847439386125368, and revolt threshold 0.7966619477648699
    Hi, I am Individual 518 with opinion 0.9215533706242405, resistance 0.6224170061470569, influence 1.1093290922695576, and revolt threshold 0.877875119850973
    Hi, I am Individual 235 with opinion -0.4838322152069858, resistance 0.5939470327490208, influence 1.2226451616623795, and revolt threshold 0.9695503044242856
    Hi, I am Individual 259 with opinion -0.6210173353560999, resistance 0.6755766731445387, influence 1.248953011698719, and revolt threshold 0.7802218720341336
    Hi, I am Individual 997 with opinion -0.4399524514878257, resistance 0.0437328241109225, influence 0.8676047426412778, and revolt threshold 0.7075174456357523
    Hi, I am Individual 715 with opinion 0.44733422321412375, resistance 0.6488040473767588, influence 0.832472440439408, and revolt threshold 0.8585847001162747
    Hi, I am Individual 442 with opinion 0.8207670144717638, resistance 0.4054427255893308, influence 0.820045426876435, and revolt threshold 0.866341721095493
    Hi, I am Individual 682 with opinion -0.3751795468292811, resistance 0.6039873179940011, influence 1.0184490225281075, and revolt threshold 0.7716192027478949
    Hi, I am Individual 588 with opinion -0.3599062574734715, resistance 0.19084098299912544, influence 1.3768055788850697, and revolt threshold 0.7886832484804692
    Hi, I am Individual 38 with opinion 0.3929256876241287, resistance 0.9594506019242242, influence 1.4110516749767408, and revolt threshold 0.7551194149010501
    Hi, I am Individual 794 with opinion -0.8694268969464327, resistance 0.2441579640653282, influence 1.3090768713084087, and revolt threshold 0.8335347152088284
    Hi, I am Individual 58 with opinion 0.25270050597954063, resistance 0.5336266098704648, influence 1.0733797653993637, and revolt threshold 0.8339991913189803
    Hi, I am Individual 185 with opinion 0.12204647844170524, resistance 0.09495679834628734, influence 1.4171362249921144, and revolt threshold 0.9723932513184899
    Hi, I am Individual 567 with opinion 0.8174318171999575, resistance 0.07632700122813763, influence 1.1766129333988258, and revolt threshold 0.7202921336724627
    Hi, I am Individual 412 with opinion -0.09517588731221238, resistance 0.41180549667146615, influence 1.3724435244665467, and revolt threshold 0.7847993651193166
    Hi, I am Individual 383 with opinion 0.06674029658965974, resistance 0.6708190463409284, influence 1.3991759421847805, and revolt threshold 0.7331917032511311
    Hi, I am Individual 165 with opinion -0.6838797791316502, resistance 0.5442109390386796, influence 1.2120013267076148, and revolt threshold 0.9549707021306361
    Hi, I am Individual 435 with opinion -0.6334098467447602, resistance 0.8041829109919882, influence 1.0910437323209765, and revolt threshold 0.8420482525162349
    Hi, I am Individual 849 with opinion 0.5865829611384521, resistance 0.06922731488617928, influence 1.4456153996571985, and revolt threshold 0.7776559714780863
    Hi, I am Individual 370 with opinion -0.8189533962141657, resistance 0.2667081639450325, influence 0.8337087567348642, and revolt threshold 0.9745330535249233
    Hi, I am Individual 695 with opinion 0.17172993038423612, resistance 0.78381156547823, influence 1.392362696883572, and revolt threshold 0.9466927597437674
    Hi, I am Individual 640 with opinion -0.9463470713247557, resistance 0.9539910876645388, influence 1.4043440177205229, and revolt threshold 0.8902238402267486
    Hi, I am Individual 922 with opinion 0.0222095742802646, resistance 0.7201481858882917, influence 1.4213086127293886, and revolt threshold 0.7033307022752872
    Hi, I am Individual 350 with opinion -0.8144237212866592, resistance 0.5719156437151647, influence 0.8764153712205794, and revolt threshold 0.8064236921296263
    Hi, I am Individual 445 with opinion -0.720677004855202, resistance 0.9629649111227316, influence 1.3239622028657196, and revolt threshold 0.8742471417787192
    Hi, I am Individual 360 with opinion 0.6915916520761636, resistance 0.46198991746399887, influence 1.3628422059003615, and revolt threshold 0.9125832532732113
    Hi, I am Individual 661 with opinion 0.85662934927544, resistance 0.5436067222314844, influence 1.0348951136185476, and revolt threshold 0.8551330117223301
    Hi, I am Individual 468 with opinion -0.6480063898489228, resistance 0.38101239713773594, influence 1.2847957793962392, and revolt threshold 0.9384769595290042
    Hi, I am Individual 426 with opinion -0.77741751922125, resistance 0.8186266620435138, influence 1.2911668331471309, and revolt threshold 0.8096812124658992
    Hi, I am an propaganda agent, you can call me Propaganda Hub 1036 and I have the id of 1036.       I have the opinion of 1, and influence strength is 0.49147134918201596      with a radius of 64.8779586687152.
    Hi, I am Individual 948 with opinion 0.5751381226513155, resistance 0.5951169838875389, influence 1.2892649349263015, and revolt threshold 0.7183495066297191
    Hi, I am Individual 836 with opinion -0.41674842971245574, resistance 0.6573539784923422, influence 1.0681499163960075, and revolt threshold 0.7375839164728869
    Hi, I am Individual 566 with opinion 0.8078561591086062, resistance 0.6615227507242721, influence 1.4217430324165146, and revolt threshold 0.8140091764218519
    Hi, I am Individual 561 with opinion 0.06909329310067869, resistance 0.45791799630934416, influence 1.4375307598965599, and revolt threshold 0.8721675125021152
    Hi, I am Individual 465 with opinion 0.858634258432992, resistance 0.8100027104753098, influence 1.2370547575037567, and revolt threshold 0.8300519134111589
    Hi, I am an propaganda agent, you can call me Propaganda Hub 1006 and I have the id of 1006.       I have the opinion of -1, and influence strength is 0.7216113538082305      with a radius of 36.22002072370366.
    Hi, I am Individual 54 with opinion -0.32090759116681755, resistance 0.15025388378203708, influence 0.9756880772132266, and revolt threshold 0.8406850564355624
    Hi, I am Individual 717 with opinion -0.4767811991982751, resistance 0.5506898788424619, influence 0.8507694900212243, and revolt threshold 0.8422742728179904
    Hi, I am Individual 218 with opinion -0.4513795099340956, resistance 0.9881173364390132, influence 1.0218302554940828, and revolt threshold 0.8678600451846161
    Hi, I am Individual 615 with opinion -0.1971436747340376, resistance 0.20393968319862754, influence 1.4270416098987613, and revolt threshold 0.9197583693556424
    Hi, I am Individual 814 with opinion 0.9408581695435472, resistance 0.6242575368947609, influence 1.4917421183307624, and revolt threshold 0.9110612412052481
    Hi, I am Individual 638 with opinion -0.38097218460301385, resistance 0.8836244354364599, influence 1.0357959886335897, and revolt threshold 0.8011163134230195
    Hi, I am Individual 20 with opinion -0.47463160793723524, resistance 0.13152231915234303, influence 1.4231340930978214, and revolt threshold 0.9546241233514479
    Hi, I am Individual 611 with opinion 0.5895304540131336, resistance 0.32749494840512294, influence 1.0334206043054133, and revolt threshold 0.963122476408051
    Hi, I am Individual 743 with opinion -0.25851181639373966, resistance 0.9244922903864154, influence 1.4219493729712376, and revolt threshold 0.9434372074035214
    Hi, I am Individual 720 with opinion -0.1985795673888413, resistance 0.5613081695862379, influence 1.3791754657870237, and revolt threshold 0.8778101613770588
    Hi, I am Individual 853 with opinion -0.7765048402671897, resistance 0.9098516009529957, influence 0.8478009684616694, and revolt threshold 0.9477713940713066
    Hi, I am Individual 989 with opinion -0.5539224669071379, resistance 0.4341190333969124, influence 0.9551391391889581, and revolt threshold 0.7165052172773546
    Hi, I am Individual 947 with opinion 0.5237610601580434, resistance 0.8385235745688319, influence 1.0208347934201485, and revolt threshold 0.7697922901116786
    Hi, I am Individual 279 with opinion -0.3251406844007476, resistance 0.3280366805335153, influence 1.4317738079837132, and revolt threshold 0.7583820365828633
    Hi, I am Individual 451 with opinion -0.5970570078923503, resistance 0.8669733066284845, influence 1.086790887081552, and revolt threshold 0.9695035417421715
    Hi, I am Individual 421 with opinion -0.07109161297286004, resistance 0.45994530088729, influence 1.1982979731274432, and revolt threshold 0.8033422276162188
    Hi, I am Individual 982 with opinion -0.05277243197587911, resistance 0.6861974514801581, influence 1.0906838610134422, and revolt threshold 0.9262945486238614
    Hi, I am Individual 923 with opinion -0.02536283026669839, resistance 0.9683038662948152, influence 1.163203348875033, and revolt threshold 0.8954318798032631
    Hi, I am Individual 386 with opinion -0.5606347419067239, resistance 0.7527730753018799, influence 1.253872877247415, and revolt threshold 0.7401388207932367
    Hi, I am Individual 49 with opinion -0.9612478625090439, resistance 0.5107625377400559, influence 0.8819714001435353, and revolt threshold 0.9303718124031852
    Hi, I am Individual 158 with opinion 0.9317759136755779, resistance 0.8538968871677026, influence 1.1486179941967476, and revolt threshold 0.9122178724791794
    Hi, I am Individual 99 with opinion 0.9552302913786297, resistance 0.9073717175023447, influence 1.3348617029784644, and revolt threshold 0.9858275282819559
    Hi, I am Individual 53 with opinion -0.4132609731416017, resistance 0.30785018710721435, influence 1.4702680988754866, and revolt threshold 0.8191117808021761
    Hi, I am Individual 388 with opinion -0.018143168034445045, resistance 0.5206056838300379, influence 1.4489145506462724, and revolt threshold 0.8888987745825645
    Hi, I am Individual 125 with opinion -0.5813926256990483, resistance 0.5650383404073772, influence 1.324652116343426, and revolt threshold 0.833705952046091
    Hi, I am Individual 198 with opinion -0.8362878892558776, resistance 0.4284644505864713, influence 1.3553490807470543, and revolt threshold 0.8476309033282242
    Hi, I am Individual 906 with opinion -0.24335766849792484, resistance 0.09768899099439599, influence 1.198094436078759, and revolt threshold 0.9861933975642609
    Hi, I am Individual 167 with opinion -0.4501377299984084, resistance 0.561252454031877, influence 1.2424519960185465, and revolt threshold 0.7678289306344976
    Hi, I am Individual 617 with opinion 0.23292961692201675, resistance 0.3014832031978727, influence 0.9457564897405291, and revolt threshold 0.8943020783477662
    Hi, I am Individual 249 with opinion 0.5308880309916488, resistance 0.15565484963615006, influence 0.9896229812524999, and revolt threshold 0.9060885083073134
    Hi, I am Individual 438 with opinion -0.20173042116395368, resistance 0.1418792721244011, influence 1.3312477581397784, and revolt threshold 0.7170253347901486
    Hi, I am Individual 319 with opinion 0.42060204997130546, resistance 0.8186782785243111, influence 0.9511954530554282, and revolt threshold 0.8780717041284078
    Hi, I am Individual 547 with opinion 0.8199559720171974, resistance 0.7069047662621523, influence 1.1799875779240931, and revolt threshold 0.7927325532493494
    Hi, I am Individual 197 with opinion 0.46346060492949426, resistance 0.6760097816605297, influence 1.1227242955661936, and revolt threshold 0.9336111843685955
    Hi, I am Individual 414 with opinion -0.48351713549426334, resistance 0.06672249839341582, influence 1.4328865907259734, and revolt threshold 0.8294777066387647
    Hi, I am Individual 467 with opinion -0.8409466674388786, resistance 0.3253024452887694, influence 1.1126651818173865, and revolt threshold 0.837981627207116
    Hi, I am Individual 523 with opinion 0.5320851876594159, resistance 0.7771498433858413, influence 1.4243471729468231, and revolt threshold 0.8644537208182104
    Hi, I am Individual 459 with opinion 0.17726835633633486, resistance 0.2365594205848981, influence 1.2594090099966035, and revolt threshold 0.7464244059449001
    Hi, I am Individual 621 with opinion 0.38955364678945403, resistance 0.9234547083132119, influence 1.3722117369661608, and revolt threshold 0.7764838863033616
    Hi, I am Individual 246 with opinion 0.6234074547158002, resistance 0.5287146337546268, influence 0.984274156947393, and revolt threshold 0.8103480364631437
    Hi, I am Individual 109 with opinion 0.28766914093565754, resistance 0.2534675876488688, influence 0.8328272775242708, and revolt threshold 0.7587849236700182
    Hi, I am Individual 972 with opinion 0.44011610582231064, resistance 0.882459543171742, influence 1.3421761322906984, and revolt threshold 0.7013066710876522
    Hi, I am Individual 978 with opinion -0.3443291215626223, resistance 0.7795303467909956, influence 1.2966248686047752, and revolt threshold 0.9966000114111713
    Hi, I am an propaganda agent, you can call me Propaganda Hub 1012 and I have the id of 1012.       I have the opinion of 1, and influence strength is 0.4957691056034795      with a radius of 77.3158482468597.
    Hi, I am Individual 690 with opinion -0.4276036290189924, resistance 0.45310482539588826, influence 1.042996845235804, and revolt threshold 0.8345449897693934
    Hi, I am Individual 6 with opinion 0.8319841474660761, resistance 0.8564755337661468, influence 1.0250145475719303, and revolt threshold 0.9979072857253499
    Hi, I am Individual 327 with opinion 0.3161698995506448, resistance 0.7078250552669544, influence 1.3292656452902774, and revolt threshold 0.8253288335775847
    Hi, I am Individual 257 with opinion 0.03921069227663665, resistance 0.22133408302006374, influence 1.0004405313778681, and revolt threshold 0.9698406328840683
    Hi, I am Individual 826 with opinion -0.21209727454673777, resistance 0.4063626968054399, influence 0.837475099664742, and revolt threshold 0.789938892591295
    Hi, I am Individual 980 with opinion 0.7649887448065471, resistance 0.9859899616009613, influence 1.0739744632777863, and revolt threshold 0.9048485756327334
    Hi, I am an propaganda agent, you can call me Propaganda Hub 1014 and I have the id of 1014.       I have the opinion of 1, and influence strength is 0.811577623077563      with a radius of 25.39281890586353.
    Hi, I am Individual 444 with opinion -0.052978226112076365, resistance 0.16786521916923192, influence 1.0069059903606947, and revolt threshold 0.8576057262679484
    Hi, I am Individual 69 with opinion -0.7801814941765177, resistance 0.7471280206862796, influence 1.118751178580913, and revolt threshold 0.9568126405569077
    Hi, I am Individual 969 with opinion -0.8619324509630129, resistance 0.9387790340828615, influence 1.2117872725051568, and revolt threshold 0.9075006220586774
    Hi, I am an propaganda agent, you can call me Propaganda Hub 1038 and I have the id of 1038.       I have the opinion of -1, and influence strength is 0.49909328486789467      with a radius of 80.7840594570704.
    Hi, I am Individual 73 with opinion 0.9149452139952075, resistance 0.2754215198637807, influence 1.4727142144878504, and revolt threshold 0.726555445404704
    Hi, I am Individual 992 with opinion -0.17243891575736936, resistance 0.1786457544359874, influence 1.2597545633049434, and revolt threshold 0.7166551878504169
    Hi, I am Individual 963 with opinion -0.9636928739441335, resistance 0.07513472627829887, influence 1.0970804439701924, and revolt threshold 0.8882395802823767
    Hi, I am Individual 825 with opinion -0.7858612782071523, resistance 0.6621362650005881, influence 1.3593275518809458, and revolt threshold 0.9529577619224248
    Hi, I am Individual 25 with opinion 0.9900508561576342, resistance 0.5946562053948186, influence 1.4707047672429763, and revolt threshold 0.7045774364318828
    Hi, I am Individual 864 with opinion -0.6313929491780541, resistance 0.1634616301684083, influence 0.9255688783109848, and revolt threshold 0.7262540060684282
    Hi, I am Individual 530 with opinion 0.5734460288363856, resistance 0.11757690142693689, influence 1.2627660417323587, and revolt threshold 0.8285729007904492
    Hi, I am Individual 145 with opinion 0.054988925746143336, resistance 0.630017947115504, influence 1.2625535409972908, and revolt threshold 0.8577205132244278
    Hi, I am Individual 42 with opinion 0.7905390545056203, resistance 0.16087486794731898, influence 1.4883460508868902, and revolt threshold 0.8511601782684501
    Hi, I am Individual 295 with opinion 0.08637991797840883, resistance 0.7563047244311433, influence 0.8894948588848418, and revolt threshold 0.7569278179728202
    Hi, I am Individual 239 with opinion 0.12064764561670005, resistance 0.040711815292841624, influence 1.4493546157997659, and revolt threshold 0.9157303987496224
    Hi, I am Individual 774 with opinion -0.7926637064076674, resistance 0.4033599221931671, influence 1.076569783068482, and revolt threshold 0.96215762349362
    Hi, I am Individual 224 with opinion 0.2853522676089024, resistance 0.3698484044854541, influence 1.3123882668783717, and revolt threshold 0.9143038440950912
    Hi, I am Individual 765 with opinion -0.10103410905314081, resistance 0.2780675144620819, influence 0.8716742690784065, and revolt threshold 0.865557308870799
    Hi, I am Individual 685 with opinion -0.28846187368932585, resistance 0.8520641518999553, influence 1.441923362667636, and revolt threshold 0.8474816472871656
    Hi, I am Individual 269 with opinion 0.11975543780608233, resistance 0.40641981678817196, influence 1.4282591682884263, and revolt threshold 0.8539689833225791
    Hi, I am Individual 785 with opinion -0.08237297295962587, resistance 0.7427019192506658, influence 0.8968206858761723, and revolt threshold 0.7366640689234022
    Hi, I am Individual 2 with opinion 0.005048954186233301, resistance 0.2715434016013998, influence 1.3326609926317858, and revolt threshold 0.7316082087882769
    Hi, I am Individual 433 with opinion 0.035300291716744425, resistance 0.8889451807942849, influence 0.8650250569854327, and revolt threshold 0.9195496415568212
    Hi, I am an propaganda agent, you can call me Propaganda Hub 1019 and I have the id of 1019.       I have the opinion of 1, and influence strength is 0.9444267413597406      with a radius of 17.07769043571401.
    Hi, I am Individual 390 with opinion 0.5234474989833906, resistance 0.6874895599094851, influence 1.399767190083561, and revolt threshold 0.7992917046435944
    Hi, I am Individual 50 with opinion -0.5018438574905324, resistance 0.4678771363438148, influence 0.8629797942449887, and revolt threshold 0.7742718376367257
    Hi, I am Individual 155 with opinion -0.054608376440219875, resistance 0.685678527731912, influence 1.250101805106766, and revolt threshold 0.7397103683909836
    Hi, I am Individual 702 with opinion -0.06134765846585144, resistance 0.5502476410540473, influence 1.0590004568831175, and revolt threshold 0.7908521220927648
    Hi, I am Individual 29 with opinion -0.18425306983907808, resistance 0.7716563920491829, influence 0.910204710642643, and revolt threshold 0.9310225648375482
    Hi, I am an propaganda agent, you can call me Propaganda Hub 1002 and I have the id of 1002.       I have the opinion of -1, and influence strength is 0.7093372800629842      with a radius of 35.213826452044685.
    Hi, I am Individual 914 with opinion 0.782067425299275, resistance 0.21856171991291695, influence 1.346109373282963, and revolt threshold 0.8857574146797632
    Hi, I am Individual 727 with opinion -0.5781441293651581, resistance 0.06579557345220999, influence 0.880727116781073, and revolt threshold 0.7638969227365731
    Hi, I am Individual 87 with opinion 0.6476315394386196, resistance 0.6422552499792191, influence 0.8859874825010059, and revolt threshold 0.965882608866465
    Hi, I am Individual 509 with opinion 0.8215444134624752, resistance 0.5732414665411146, influence 1.418252084255637, and revolt threshold 0.8146513591653928
    Hi, I am Individual 302 with opinion -0.3823439801737596, resistance 0.9234278951836236, influence 0.9535356186974536, and revolt threshold 0.7656703218898884
    Hi, I am Individual 152 with opinion 0.4154081590410643, resistance 0.49418276597121413, influence 1.0416524514618832, and revolt threshold 0.8999470011309917
    Hi, I am Individual 865 with opinion 0.3426548253035857, resistance 0.9794500898230051, influence 1.1841264274983332, and revolt threshold 0.7189008269741486
    Hi, I am Individual 507 with opinion 0.03094621595431213, resistance 0.9877982186052986, influence 0.8739126251866224, and revolt threshold 0.9720157799169806
    Hi, I am Individual 686 with opinion 0.8108211131378897, resistance 0.09181815867123788, influence 1.0987672907275157, and revolt threshold 0.7011503944481691
    Hi, I am Individual 488 with opinion -0.2020841747607678, resistance 0.2506781717156574, influence 0.9118030602229068, and revolt threshold 0.9945714393162569
    Hi, I am an propaganda agent, you can call me Propaganda Hub 1008 and I have the id of 1008.       I have the opinion of 1, and influence strength is 0.8676820503248781      with a radius of 75.53616084288922.
    Hi, I am Individual 833 with opinion -0.10725345535835795, resistance 0.9982921971445897, influence 1.270276918670762, and revolt threshold 0.8923216835636643
    Hi, I am Individual 555 with opinion 0.14066691734154646, resistance 0.5615975778114116, influence 1.351036711745179, and revolt threshold 0.7540491744768238
    Hi, I am Individual 494 with opinion -0.7068263942125568, resistance 0.8397208475861417, influence 1.458615919255676, and revolt threshold 0.9617832675657156
    Hi, I am Individual 486 with opinion -0.16164117420133617, resistance 0.42799590485884054, influence 1.351822135860319, and revolt threshold 0.9233260182870691
    Hi, I am Individual 359 with opinion 0.35670286590459677, resistance 0.844185660696777, influence 0.8170416828908171, and revolt threshold 0.9760565788166455
    Hi, I am Individual 331 with opinion 0.5544885015382404, resistance 0.8429127805176146, influence 1.0911188011583701, and revolt threshold 0.7491255740996684
    Hi, I am Individual 527 with opinion -0.3022792306503357, resistance 0.3785389625737495, influence 1.431473033914652, and revolt threshold 0.7619160862502659
    Hi, I am Individual 885 with opinion -0.33487392396515014, resistance 0.3388964410376287, influence 1.2145336700621385, and revolt threshold 0.7497640905708894
    Hi, I am Individual 859 with opinion -0.5322446119119844, resistance 0.6553326547291215, influence 0.9442201009472845, and revolt threshold 0.9713106098633825
    Hi, I am Individual 345 with opinion 0.17854850573935654, resistance 0.7731000923242152, influence 1.2824092440554877, and revolt threshold 0.7052710361802471
    Hi, I am Individual 548 with opinion 0.6319302605158224, resistance 0.9038499916030293, influence 1.3632484254983466, and revolt threshold 0.7915856054405549
    Hi, I am Individual 960 with opinion 0.22440138975209822, resistance 0.20400453929777462, influence 1.254210020884362, and revolt threshold 0.9894022595343042
    Hi, I am Individual 805 with opinion -0.21953028062342583, resistance 0.8870225754251692, influence 1.0917453167056115, and revolt threshold 0.7195518213923583
    Hi, I am Individual 206 with opinion -0.7886499656039747, resistance 0.6556017964328386, influence 1.2993795279152676, and revolt threshold 0.9421868618778664
    Hi, I am Individual 594 with opinion -0.5468319053567121, resistance 0.5072170751470543, influence 1.4376427815006811, and revolt threshold 0.771461658890445
    Hi, I am Individual 912 with opinion 0.5487494757102351, resistance 0.01449101622647464, influence 1.299541035665699, and revolt threshold 0.862648433545322
    Hi, I am Individual 289 with opinion 0.10860635118105622, resistance 0.8201801057174404, influence 0.9951745805892255, and revolt threshold 0.9058904079067351
    Hi, I am Individual 234 with opinion 0.48216844164177974, resistance 0.8608476665788302, influence 1.4013472166043281, and revolt threshold 0.9871320900460501
    Hi, I am Individual 231 with opinion -0.8457246129411262, resistance 0.9236334993294741, influence 1.252586893552838, and revolt threshold 0.8194622223406928
    Hi, I am an propaganda agent, you can call me Propaganda Hub 1004 and I have the id of 1004.       I have the opinion of -1, and influence strength is 0.8715868200284234      with a radius of 88.1673176820161.
    Hi, I am Individual 470 with opinion 0.9192341373057, resistance 0.013724833926678337, influence 1.114379294935286, and revolt threshold 0.7725544209302326
    Hi, I am Individual 100 with opinion -0.3360109931376767, resistance 0.7671265184323561, influence 1.4215124899908027, and revolt threshold 0.8052831645595679
    Hi, I am Individual 525 with opinion 0.43252418246010715, resistance 0.8723662939278609, influence 0.8288991557404161, and revolt threshold 0.8816148648155209
    Hi, I am Individual 263 with opinion -0.22754011694101384, resistance 0.24978188605891627, influence 0.9099629462895689, and revolt threshold 0.7410944799669921
    Hi, I am Individual 482 with opinion 0.06759950262524295, resistance 0.21348690976786822, influence 0.9564922679450762, and revolt threshold 0.7220875443943839
    Hi, I am Individual 641 with opinion 0.0849153826144029, resistance 0.07159619960008556, influence 0.9285902825036921, and revolt threshold 0.9890740343766213
    Hi, I am Individual 779 with opinion 0.48700232063705684, resistance 0.48804013635724275, influence 1.0298113756460723, and revolt threshold 0.7227692423997726
    Hi, I am Individual 535 with opinion 0.759250465562548, resistance 0.29240060808622526, influence 1.3876991307044557, and revolt threshold 0.8696685788619978
    Hi, I am Individual 506 with opinion 0.9609100099603005, resistance 0.11690117862733207, influence 1.4749051202810168, and revolt threshold 0.9151239332695208
    Hi, I am Individual 298 with opinion -0.4842396161502407, resistance 0.7292189868863111, influence 1.3505248847021523, and revolt threshold 0.8363462192352564
    Hi, I am Individual 82 with opinion 0.7822003285042842, resistance 0.17496064577233028, influence 1.2046132074285107, and revolt threshold 0.9925610564661798
    Hi, I am Individual 753 with opinion -0.9734154362533975, resistance 0.32494100146883265, influence 1.235871821250481, and revolt threshold 0.780722540985104
    Hi, I am Individual 564 with opinion -0.7641635568595859, resistance 0.6657094371542298, influence 1.2408489046913935, and revolt threshold 0.8016561901009462
    Hi, I am Individual 291 with opinion -0.4361562548689808, resistance 0.8600183387772092, influence 0.8113226355092201, and revolt threshold 0.7629554122624514
    Hi, I am Individual 156 with opinion -0.13367788902820688, resistance 0.09041131515507295, influence 1.128267113753568, and revolt threshold 0.7520465608351273
    Hi, I am an propaganda agent, you can call me Propaganda Hub 1022 and I have the id of 1022.       I have the opinion of 1, and influence strength is 0.46474649552219405      with a radius of 98.4645468139636.
    Hi, I am Individual 962 with opinion -0.3264871048072129, resistance 0.7113282371289756, influence 1.484117741645461, and revolt threshold 0.795435136766762
    Hi, I am Individual 736 with opinion 0.3985872385969771, resistance 0.739933935528026, influence 0.8137370433252921, and revolt threshold 0.8142713749495832
    Hi, I am Individual 261 with opinion 0.9782182577440968, resistance 0.9823031288144282, influence 0.8733719039717288, and revolt threshold 0.7493991631504996
    Hi, I am Individual 881 with opinion -0.0883053946155199, resistance 0.5824532710811966, influence 1.261137101736609, and revolt threshold 0.978968961502394
    Hi, I am an propaganda agent, you can call me Propaganda Hub 1047 and I have the id of 1047.       I have the opinion of -1, and influence strength is 0.9921598013525055      with a radius of 36.110504799954924.
    Hi, I am Individual 901 with opinion -0.4354863274843137, resistance 0.7972076660927104, influence 1.0430126417932102, and revolt threshold 0.7527788135704614
    Hi, I am Individual 910 with opinion -0.16634683836545872, resistance 0.7014655471565302, influence 1.1881513136350232, and revolt threshold 0.778697011594102
    Hi, I am Individual 735 with opinion -0.7296288601346617, resistance 0.7788316405213698, influence 1.187510750043136, and revolt threshold 0.8194925417586248
    Hi, I am Individual 307 with opinion 0.9306139646907441, resistance 0.4695496177041534, influence 1.0235342158822107, and revolt threshold 0.8295781987005104
    Hi, I am Individual 784 with opinion -0.18359994744582142, resistance 0.08494274283696124, influence 1.1296956742639144, and revolt threshold 0.8252721735772403
    Hi, I am Individual 604 with opinion -0.01572821536821789, resistance 0.1503350197700355, influence 1.009447292586921, and revolt threshold 0.7082819225737912
    Hi, I am Individual 810 with opinion -0.054736427265783316, resistance 0.5519564772774291, influence 1.114441289852136, and revolt threshold 0.9468689356440543
    Hi, I am an propaganda agent, you can call me Propaganda Hub 1018 and I have the id of 1018.       I have the opinion of -1, and influence strength is 0.8459221508330244      with a radius of 53.13073122286961.
    Hi, I am Individual 323 with opinion -0.36350877112138225, resistance 0.904518236411327, influence 0.9524474039545279, and revolt threshold 0.873064335801906
    Hi, I am Individual 586 with opinion 0.1403616814980826, resistance 0.11252425944251176, influence 1.1388952507858305, and revolt threshold 0.9458184755960874
    Hi, I am Individual 935 with opinion 0.20104524407815938, resistance 0.29868117515543446, influence 1.2666583389282082, and revolt threshold 0.7255054465244307
    Hi, I am Individual 462 with opinion -0.27158452452483806, resistance 0.23243392788957706, influence 1.4854211691715542, and revolt threshold 0.8186923252167504
    Hi, I am Individual 620 with opinion -0.26009423562822653, resistance 0.4911414694695222, influence 1.269621231932545, and revolt threshold 0.8592716087564396
    Hi, I am Individual 324 with opinion -0.07094310168073714, resistance 0.5934218086245079, influence 1.4308169038266687, and revolt threshold 0.9221670031165453
    Hi, I am Individual 930 with opinion -0.9920048431013617, resistance 0.8576973250999084, influence 1.0454416839156542, and revolt threshold 0.7497740853505775
    Hi, I am Individual 722 with opinion -0.9422789610465689, resistance 0.4303934354448373, influence 0.8784957682058927, and revolt threshold 0.8865238816874395
    Hi, I am Individual 376 with opinion -0.12800798859050344, resistance 0.9322672382517019, influence 1.064948390390727, and revolt threshold 0.7103389884669066
    Hi, I am Individual 719 with opinion -0.915790886869043, resistance 0.9495479236108475, influence 1.001082891746619, and revolt threshold 0.841203612809708
    Hi, I am Individual 772 with opinion -0.9166026471671924, resistance 0.9007211672149211, influence 1.446510691476891, and revolt threshold 0.7593038349760481
    Hi, I am Individual 816 with opinion -0.5446150050532734, resistance 0.13445859317751407, influence 1.2365725673115406, and revolt threshold 0.7151651754330477
    Hi, I am Individual 798 with opinion -0.22532379144364523, resistance 0.3308357831430859, influence 1.0592588631917863, and revolt threshold 0.8585281148655235
    Hi, I am Individual 314 with opinion 0.17735433597587202, resistance 0.865487019181973, influence 1.3847760001079936, and revolt threshold 0.9585906410745035
    Hi, I am Individual 524 with opinion -0.37063445391228633, resistance 0.11406148040465047, influence 0.9588422682820593, and revolt threshold 0.9716022576181522
    Hi, I am Individual 987 with opinion -0.12712720458720583, resistance 0.003054727525657608, influence 1.2890611131726422, and revolt threshold 0.988864697955381
    Hi, I am Individual 920 with opinion 0.08335789357374135, resistance 0.48807097112221054, influence 1.3158564347407118, and revolt threshold 0.8095180175192103
    Hi, I am Individual 954 with opinion 0.010695813837344526, resistance 0.0963674686235757, influence 1.4205314499620516, and revolt threshold 0.7262819142741819
    Hi, I am Individual 160 with opinion 0.06320983993665896, resistance 0.10196104591554322, influence 1.3949163370861437, and revolt threshold 0.8163118628427506
    Hi, I am Individual 275 with opinion 0.25766160035644337, resistance 0.5531168990576439, influence 1.1473624893702588, and revolt threshold 0.7942007798186038
    Hi, I am an propaganda agent, you can call me Propaganda Hub 1035 and I have the id of 1035.       I have the opinion of 1, and influence strength is 0.9798799212716517      with a radius of 55.39755572234994.
    Hi, I am Individual 984 with opinion -0.30545850143994646, resistance 0.18929619768607853, influence 1.2835777446191192, and revolt threshold 0.8499634208900162
    Hi, I am Individual 707 with opinion -0.2862436503846548, resistance 0.38339042946439905, influence 1.068077175022946, and revolt threshold 0.8162504710144477
    Hi, I am Individual 236 with opinion -0.022570117204125273, resistance 0.5665110588329475, influence 1.269754845607272, and revolt threshold 0.8389401876682925
    Hi, I am Individual 418 with opinion 0.962797692767241, resistance 0.5217012818771772, influence 1.1445458916590854, and revolt threshold 0.8728046618319419
    Hi, I am Individual 605 with opinion -0.02204135799464013, resistance 0.5242848402137077, influence 0.8208789502653973, and revolt threshold 0.8046264096661055
    Hi, I am Individual 475 with opinion 0.0661191632469158, resistance 0.8766896177838617, influence 0.9716849051533042, and revolt threshold 0.8042982395686407
    Hi, I am Individual 803 with opinion 0.01563254015935378, resistance 0.47212773806482655, influence 1.1719843377356778, and revolt threshold 0.7792207075570968
    Hi, I am Individual 625 with opinion 0.6858188322009136, resistance 0.675179029081384, influence 1.325501687592388, and revolt threshold 0.7663563791037099
    Hi, I am Individual 454 with opinion -0.5829167671981517, resistance 0.4001927058485235, influence 1.3426779173522974, and revolt threshold 0.8881522487160101
    Hi, I am Individual 603 with opinion -0.968707728663835, resistance 0.8190048208694958, influence 1.136181156545952, and revolt threshold 0.9195892964265885
    Hi, I am Individual 505 with opinion 0.9946621614596578, resistance 0.508814238731239, influence 0.921205191720128, and revolt threshold 0.7656328460896672
    Hi, I am Individual 755 with opinion 0.7847980278001989, resistance 0.4365712558972431, influence 0.9789503253250718, and revolt threshold 0.7938754615707092
    Hi, I am Individual 51 with opinion -0.1530232617367644, resistance 0.40109283542388974, influence 1.14527743954157, and revolt threshold 0.7895303598314916
    Hi, I am Individual 861 with opinion -0.9658195260536668, resistance 0.08870831753249697, influence 1.0423613227351956, and revolt threshold 0.7596451566719903
    Hi, I am an propaganda agent, you can call me Propaganda Hub 1044 and I have the id of 1044.       I have the opinion of -1, and influence strength is 0.4077903999790346      with a radius of 49.28574091124594.
    Hi, I am Individual 208 with opinion 0.059474942806422515, resistance 0.6744346286798203, influence 0.9614152491263444, and revolt threshold 0.7186810370290314
    Hi, I am Individual 706 with opinion -0.8041923989084554, resistance 0.5528551490688831, influence 0.8291459158416847, and revolt threshold 0.98787512855559
    Hi, I am Individual 678 with opinion 0.6341573997610639, resistance 0.15121700862503462, influence 1.3898605839406573, and revolt threshold 0.7612381558855558
    Hi, I am Individual 446 with opinion -0.14318068883917734, resistance 0.8460134160264436, influence 1.1908067331180916, and revolt threshold 0.9065832303551633
    Hi, I am Individual 894 with opinion 0.735321145413036, resistance 0.4434799912959396, influence 0.9951474654863728, and revolt threshold 0.980255725887576
    Hi, I am Individual 662 with opinion -0.5745499979732274, resistance 0.44236498524723966, influence 1.094157414341531, and revolt threshold 0.7592466125632412
    Hi, I am an propaganda agent, you can call me Propaganda Hub 1020 and I have the id of 1020.       I have the opinion of 1, and influence strength is 0.4920076917617785      with a radius of 50.237073982242386.
    Hi, I am Individual 453 with opinion -0.41680876144186674, resistance 0.2830954961059772, influence 0.9216460211700201, and revolt threshold 0.8596706165193242
    Hi, I am Individual 361 with opinion -0.9009825532646767, resistance 0.11963451503165867, influence 0.9628720628216024, and revolt threshold 0.7593732084372086
    Hi, I am Individual 334 with opinion -0.3722648245654139, resistance 0.5756597014172945, influence 1.0592634341728522, and revolt threshold 0.760700033702099
    Hi, I am Individual 273 with opinion 0.3950028251380724, resistance 0.5113016153916043, influence 1.1791176698821038, and revolt threshold 0.9933882252272236
    Hi, I am Individual 716 with opinion -0.46818770558307077, resistance 0.13004604414418675, influence 0.9147588630198498, and revolt threshold 0.7563068873987727
    Hi, I am Individual 515 with opinion 0.8817471171682605, resistance 0.6601333029985041, influence 0.9038321806729487, and revolt threshold 0.9643938102564042
    Hi, I am Individual 3 with opinion 0.15739834369014738, resistance 0.028778564988380273, influence 0.9773828681430491, and revolt threshold 0.7012409204500716
    Hi, I am Individual 65 with opinion 0.025083669936567965, resistance 0.3938148582277897, influence 1.061613059667191, and revolt threshold 0.814908700120083
    Hi, I am Individual 357 with opinion -0.7795737119463526, resistance 0.8347613596937772, influence 1.1025295481529687, and revolt threshold 0.8322445936374799
    Hi, I am Individual 35 with opinion 0.5303966655116719, resistance 0.9118980042284195, influence 0.8110716419521486, and revolt threshold 0.9258856137835692
    Hi, I am Individual 592 with opinion 0.793356502496527, resistance 0.9165176270121719, influence 1.416262829881096, and revolt threshold 0.9563541983424755
    Hi, I am Individual 378 with opinion 0.8402248043675005, resistance 0.711865310126407, influence 1.0255838323513857, and revolt threshold 0.9741216261641579
    Hi, I am Individual 934 with opinion -0.13686703717352255, resistance 0.6102701363910148, influence 1.3241677168324286, and revolt threshold 0.9862331540471652
    Hi, I am Individual 64 with opinion -0.45127533280205534, resistance 0.37690226370750113, influence 1.1196305373947915, and revolt threshold 0.9586355828163411
    Hi, I am Individual 424 with opinion 0.8903512288022553, resistance 0.20587500108620516, influence 0.939813641873603, and revolt threshold 0.9980240694022675
    Hi, I am Individual 634 with opinion -0.19858260395282623, resistance 0.22055698082648367, influence 1.3839741227736986, and revolt threshold 0.9887656059033626
    Hi, I am Individual 823 with opinion 0.8175753771560528, resistance 0.4298107176874756, influence 1.0482796482609353, and revolt threshold 0.911904525188475
    Hi, I am Individual 402 with opinion 0.966752512343185, resistance 0.11317727787860976, influence 1.014729445503486, and revolt threshold 0.723336887840867
    Hi, I am Individual 232 with opinion 0.5333709254741814, resistance 0.7025386187793812, influence 1.1551723884625074, and revolt threshold 0.7499718363666942
    Hi, I am Individual 758 with opinion -0.7929614832156267, resistance 0.40425466948605326, influence 0.8264841605184658, and revolt threshold 0.9976995289153507
    Hi, I am Individual 739 with opinion 0.6064439838018785, resistance 0.47056997910960385, influence 1.2227792616973239, and revolt threshold 0.7528595841333551
    Hi, I am Individual 21 with opinion -0.4765486150293463, resistance 0.21300590069585956, influence 1.2054669634094313, and revolt threshold 0.8253866920352224
    Hi, I am Individual 434 with opinion -0.7970295812240562, resistance 0.18524961237015514, influence 1.1304341629615846, and revolt threshold 0.8119158732299218
    Hi, I am Individual 626 with opinion 0.8926766888121842, resistance 0.92973124788548, influence 1.1032400748780509, and revolt threshold 0.9612592820397747
    Hi, I am Individual 733 with opinion 0.20256151662624888, resistance 0.7012387208269706, influence 1.1746615915426113, and revolt threshold 0.948946787284479
    Hi, I am Individual 13 with opinion -0.8546093165596993, resistance 0.7831175965268992, influence 1.1862141189219333, and revolt threshold 0.739115415856334
    Hi, I am Individual 822 with opinion 0.32445061922536933, resistance 0.7386372577969599, influence 1.3552737434141926, and revolt threshold 0.9836188425278516
    Hi, I am Individual 389 with opinion -0.9060923877342699, resistance 0.06793954830858007, influence 1.0643999258772707, and revolt threshold 0.8204355207179141
    Hi, I am Individual 876 with opinion 0.2378444042214778, resistance 0.34738885506382544, influence 1.011464303043688, and revolt threshold 0.7958235436441656
    Hi, I am an propaganda agent, you can call me Propaganda Hub 1010 and I have the id of 1010.       I have the opinion of 1, and influence strength is 0.8517917695381629      with a radius of 74.82169466152679.
    Hi, I am Individual 793 with opinion -0.3965423850616516, resistance 0.5763347296208141, influence 1.0663486657466934, and revolt threshold 0.9775327324489396
    Hi, I am Individual 481 with opinion -0.8054502581102758, resistance 0.6150354793302109, influence 1.0347057399961945, and revolt threshold 0.7928161942487335
    Hi, I am Individual 22 with opinion 0.9867612147390312, resistance 0.3302113645088498, influence 1.3713767055778066, and revolt threshold 0.974653847296889
    Hi, I am Individual 367 with opinion 0.6471780064358164, resistance 0.6216795277906136, influence 1.331696324566145, and revolt threshold 0.979923079496391
    Hi, I am Individual 723 with opinion -0.3486766370830625, resistance 0.4058435193240687, influence 1.2591696657143097, and revolt threshold 0.7569048827298146
    Hi, I am Individual 303 with opinion -0.4447657107670684, resistance 0.8919607133233897, influence 1.1668614511852011, and revolt threshold 0.9955801589835701
    Hi, I am Individual 194 with opinion 0.7952438142456844, resistance 0.32187743250232936, influence 1.2898614295245947, and revolt threshold 0.8431845045882934
    Hi, I am Individual 265 with opinion -0.5116269233102513, resistance 0.7833145626170857, influence 1.1208971782258712, and revolt threshold 0.9231836332740465
    Hi, I am Individual 919 with opinion 0.4850926907328541, resistance 0.29433564126634393, influence 1.2342778276940949, and revolt threshold 0.7395232632767542
    Hi, I am Individual 175 with opinion -0.2764291648561843, resistance 0.41601390655815496, influence 1.181464214513083, and revolt threshold 0.7715743177231584
    Hi, I am Individual 255 with opinion 0.7502168478799307, resistance 0.1344136472446047, influence 1.14159249068462, and revolt threshold 0.8480165923340666
    Hi, I am Individual 789 with opinion 0.5751504620425798, resistance 0.3784372927698558, influence 1.208445264222135, and revolt threshold 0.8922560861222788
    Hi, I am Individual 245 with opinion -0.5723673544452179, resistance 0.5726751348666087, influence 0.8881111424699236, and revolt threshold 0.814438734537884
    Hi, I am Individual 776 with opinion 0.4497630325980504, resistance 0.23668591538498895, influence 1.410644861826202, and revolt threshold 0.8007973326649893
    Hi, I am Individual 138 with opinion -0.8299355100126582, resistance 0.734798106344406, influence 1.4079095477382952, and revolt threshold 0.9602398897564492
    Hi, I am Individual 955 with opinion 0.38871587859982837, resistance 0.2682872430438141, influence 1.381816410976442, and revolt threshold 0.9646439532111055
    Hi, I am Individual 643 with opinion 0.952273700288413, resistance 0.6312545522883528, influence 1.2662499062101285, and revolt threshold 0.9252591365169959
    Hi, I am Individual 516 with opinion -0.3917973579482492, resistance 0.18924322952675898, influence 1.2757311087611185, and revolt threshold 0.9799273618660868
    Hi, I am Individual 336 with opinion 0.48446701278574467, resistance 0.38610661296991955, influence 1.0475308995781711, and revolt threshold 0.9226666161205469
    Hi, I am an propaganda agent, you can call me Propaganda Hub 1023 and I have the id of 1023.       I have the opinion of 1, and influence strength is 0.714306421705552      with a radius of 71.8033017538756.
    Hi, I am Individual 278 with opinion -0.322832438520555, resistance 0.2263954268340559, influence 1.4584704248248992, and revolt threshold 0.793795643148592
    Hi, I am Individual 986 with opinion -0.18353785624358276, resistance 0.74073026586114, influence 1.124245446128842, and revolt threshold 0.8000457624271178
    Hi, I am Individual 815 with opinion -0.022373987903832937, resistance 0.5333289218953043, influence 0.8464891294025352, and revolt threshold 0.9815921265858751
    Hi, I am Individual 877 with opinion 0.5617458171502274, resistance 0.019784773026624403, influence 1.3919800975862113, and revolt threshold 0.8253135697463583
    Hi, I am Individual 26 with opinion 0.8038017621819091, resistance 0.9626257291539507, influence 1.423950442446195, and revolt threshold 0.8408623518517652
    Hi, I am Individual 148 with opinion -0.25587291940697665, resistance 0.24886455561814358, influence 0.836701115298719, and revolt threshold 0.9804232420248795
    Hi, I am Individual 98 with opinion 0.5408018702129975, resistance 0.6991745888741014, influence 0.9709722832206428, and revolt threshold 0.7188411309778007
    Hi, I am Individual 356 with opinion 0.44976710586341206, resistance 0.33070742686129173, influence 1.4132740566456645, and revolt threshold 0.8033073116349194
    Hi, I am Individual 248 with opinion -0.6636007889443323, resistance 0.5805375394704376, influence 1.1413445205560235, and revolt threshold 0.8803966770258969
    Hi, I am Individual 606 with opinion 0.4712743476384633, resistance 0.9578666171561674, influence 0.898105794799794, and revolt threshold 0.7283449756062378
    Hi, I am Individual 322 with opinion -0.13851043625497672, resistance 0.22005984651154908, influence 1.4840345419268264, and revolt threshold 0.8490640403912757
    Hi, I am Individual 700 with opinion 0.6123055046663282, resistance 0.8554969903580131, influence 0.9613176185076234, and revolt threshold 0.8932370414667254
    Hi, I am Individual 782 with opinion 0.7735449950482383, resistance 0.7868861290495229, influence 1.4532042219029013, and revolt threshold 0.9748947206289458
    Hi, I am Individual 812 with opinion -0.9591415471964198, resistance 0.22308417743672204, influence 1.2145578149820537, and revolt threshold 0.9015058395866823
    Hi, I am Individual 473 with opinion 0.3246817138864704, resistance 0.34394779225304817, influence 1.0029869117273267, and revolt threshold 0.7363138651052051
    Hi, I am Individual 871 with opinion -0.4202837567307456, resistance 0.5861791488763989, influence 1.2628128836872579, and revolt threshold 0.7260456998569998
    Hi, I am Individual 801 with opinion -0.4776358111086625, resistance 0.5373994975916653, influence 1.2295547746479278, and revolt threshold 0.9538276384750923
    Hi, I am Individual 531 with opinion -0.9120929609056085, resistance 0.18589772477253663, influence 1.3858727806840414, and revolt threshold 0.9776110924834176
    Hi, I am Individual 247 with opinion -0.5013766751245872, resistance 0.9806775104391939, influence 1.3803240726885964, and revolt threshold 0.8442996350339644
    Hi, I am Individual 266 with opinion -0.46154897318180166, resistance 0.11190904224543341, influence 0.9282328245532953, and revolt threshold 0.8660395822709281
    Hi, I am Individual 773 with opinion 0.8422887531244525, resistance 0.9948258047079941, influence 1.39486172889075, and revolt threshold 0.9762812437551285
    Hi, I am Individual 495 with opinion -0.7159067892311572, resistance 0.915641959789538, influence 1.056104669801454, and revolt threshold 0.8799484892757568
    Hi, I am Individual 400 with opinion -0.2632665128557532, resistance 0.3603517722104148, influence 1.1614418774610733, and revolt threshold 0.973349542910437
    Hi, I am Individual 597 with opinion -0.876720030634023, resistance 0.4864180115732438, influence 1.451332693329416, and revolt threshold 0.9870484982663481
    Hi, I am Individual 301 with opinion -0.5311252190749245, resistance 0.7551891332383491, influence 0.9589458626493861, and revolt threshold 0.8372332279570245
    Hi, I am Individual 262 with opinion -0.09950358441646934, resistance 0.4111059627297866, influence 1.371772805186154, and revolt threshold 0.844186510112497
    Hi, I am Individual 141 with opinion 0.6465829921542774, resistance 0.8751855086946312, influence 1.4717705028011037, and revolt threshold 0.7912583888773115
    Hi, I am Individual 689 with opinion 0.8321248874788818, resistance 0.4919767409542729, influence 0.871347439155853, and revolt threshold 0.795503954278173
    Hi, I am Individual 14 with opinion -0.5570537702831335, resistance 0.554723378985717, influence 1.4932119791709169, and revolt threshold 0.7431391792364422
    Hi, I am Individual 804 with opinion -0.7180551649779068, resistance 0.2700247197530474, influence 1.2561629197895052, and revolt threshold 0.8246880490655975
    Hi, I am Individual 114 with opinion -0.3560087080306762, resistance 0.20130807844871856, influence 1.2222369236768016, and revolt threshold 0.9852442613807368
    Hi, I am Individual 854 with opinion -0.5781822331938287, resistance 0.08696035214577014, influence 1.209453114815247, and revolt threshold 0.7681633679606994
    Hi, I am Individual 484 with opinion -0.6975496050645051, resistance 0.2132835887318022, influence 1.2644002578177687, and revolt threshold 0.9068013895656183
    Hi, I am Individual 463 with opinion -0.4396984738425258, resistance 0.4764001688547512, influence 1.4323604147315396, and revolt threshold 0.8247007813705818
    Hi, I am Individual 338 with opinion 0.5362403244287655, resistance 0.8076262461075336, influence 0.9792686772294382, and revolt threshold 0.9707391381865674
    Hi, I am Individual 924 with opinion 0.9915965898332677, resistance 0.12190905212988368, influence 1.451781858468764, and revolt threshold 0.7959434535643558
    Hi, I am Individual 241 with opinion 0.25224399197589586, resistance 0.45371183828862593, influence 0.9606123717424675, and revolt threshold 0.9879225281575494
    Hi, I am Individual 142 with opinion -0.30469823694817166, resistance 0.5540945755751405, influence 1.3078272031327263, and revolt threshold 0.7739428317694766
    Hi, I am Individual 778 with opinion 0.6037207098801931, resistance 0.6949460056975134, influence 1.4048497531992192, and revolt threshold 0.9710577078503746
    Hi, I am Individual 81 with opinion 0.04858414357023455, resistance 0.7216930692568059, influence 1.4976302214172281, and revolt threshold 0.7120188730311872
    Hi, I am Individual 729 with opinion 0.7352408629281173, resistance 0.9961914309493819, influence 0.9869087527519685, and revolt threshold 0.7134031309036755
    Hi, I am Individual 845 with opinion 0.4616510066990318, resistance 0.5038296799918432, influence 1.260915073031742, and revolt threshold 0.7681114913882117
    Hi, I am Individual 636 with opinion 0.1805626317774922, resistance 0.8307640837636007, influence 1.0247196091158721, and revolt threshold 0.8114595985954351
    Hi, I am Individual 714 with opinion -0.6777114448932042, resistance 0.3148906700083365, influence 0.8603271764395366, and revolt threshold 0.8809465522487698
    Hi, I am Individual 747 with opinion -0.021360718206440366, resistance 0.2581533109916575, influence 1.1439206565776057, and revolt threshold 0.7648533164370261
    Hi, I am Individual 123 with opinion -0.3530852618486817, resistance 0.6732029995330445, influence 0.9614264425461808, and revolt threshold 0.7987686045998543
    Hi, I am Individual 170 with opinion -0.08438225589393533, resistance 0.22288665992874268, influence 1.053025619453766, and revolt threshold 0.9256083747419448
    Hi, I am an propaganda agent, you can call me Propaganda Hub 1033 and I have the id of 1033.       I have the opinion of 1, and influence strength is 0.9132473199719093      with a radius of 18.401726934417482.
    Hi, I am Individual 366 with opinion -0.012895170569119374, resistance 0.20764932875521713, influence 1.0461846698255952, and revolt threshold 0.7621088746338638
    Hi, I am Individual 276 with opinion -0.363857472568508, resistance 0.020982248399655234, influence 0.962478624639253, and revolt threshold 0.9475274354529586
    Hi, I am Individual 927 with opinion -0.6388065814563137, resistance 0.709380111376752, influence 0.8634686957408276, and revolt threshold 0.9217048667826215
    Hi, I am Individual 762 with opinion -0.15618711073319935, resistance 0.22315997071540128, influence 0.8184773883841435, and revolt threshold 0.8052991001287989
    Hi, I am Individual 128 with opinion -0.9328858859727669, resistance 0.9729337631394424, influence 1.0907434198119659, and revolt threshold 0.7123235159081708
    Hi, I am Individual 416 with opinion -0.5823056142845955, resistance 0.4956728484254601, influence 0.9295648286940272, and revolt threshold 0.7643648800701546
    Hi, I am Individual 74 with opinion 0.06322868310260743, resistance 0.07356675661155032, influence 1.0795940010074934, and revolt threshold 0.885579286697016
    Hi, I am Individual 759 with opinion -0.23790822295438563, resistance 0.3071795219699247, influence 1.4255108699438743, and revolt threshold 0.7823380223425703
    Hi, I am an propaganda agent, you can call me Propaganda Hub 1041 and I have the id of 1041.       I have the opinion of 1, and influence strength is 0.588294015115314      with a radius of 46.64376960119076.
    Hi, I am Individual 346 with opinion -0.8255082170466193, resistance 0.7442895569552032, influence 0.9962456890016973, and revolt threshold 0.9843984324417531
    Hi, I am Individual 34 with opinion 0.542616248586028, resistance 0.2204978644862775, influence 1.4100493272599643, and revolt threshold 0.827834650808948
    Hi, I am Individual 511 with opinion -0.14075774351581738, resistance 0.442223386118451, influence 1.4929253023114566, and revolt threshold 0.9031399256219064
    Hi, I am Individual 341 with opinion 0.1617345470378999, resistance 0.7534035514892186, influence 0.9347770548143526, and revolt threshold 0.7595249828107845
    Hi, I am Individual 270 with opinion -0.43751797812753956, resistance 0.6404405382908408, influence 1.1884895640894029, and revolt threshold 0.7994263712703265
    Hi, I am Individual 589 with opinion -0.5095356887416176, resistance 0.7813868745336984, influence 0.969558610926809, and revolt threshold 0.8115935602514134
    Hi, I am Individual 855 with opinion -0.3754328211708855, resistance 0.6953293775315269, influence 0.8250109626154154, and revolt threshold 0.7173258599881114
    Hi, I am Individual 696 with opinion 0.8534447625132104, resistance 0.8431208065123263, influence 0.9353006793642948, and revolt threshold 0.7370491788961286
    Hi, I am Individual 880 with opinion 0.5534073752646929, resistance 0.23585464050591132, influence 1.4812894515667585, and revolt threshold 0.916543465796061
    Hi, I am Individual 217 with opinion 0.6352548202576205, resistance 0.03989626295055615, influence 1.0593432879854328, and revolt threshold 0.7517584226694698
    Hi, I am Individual 581 with opinion -0.5268926435562844, resistance 0.08683951373582588, influence 1.0679956899641672, and revolt threshold 0.825415724060728
    Hi, I am Individual 647 with opinion -0.3466813432810223, resistance 0.4209775171258797, influence 0.871863056273145, and revolt threshold 0.8055148283455617
    Hi, I am Individual 905 with opinion 0.3346350843760506, resistance 0.5653242814427459, influence 1.3380037399631044, and revolt threshold 0.7082661472875853
    Hi, I am Individual 60 with opinion 0.7515792061093465, resistance 0.21885648205981356, influence 1.0942245127939974, and revolt threshold 0.777536264556936
    Hi, I am Individual 254 with opinion -0.9452965782018194, resistance 0.8880013607305787, influence 1.18257588766334, and revolt threshold 0.8521165812354814
    Hi, I am Individual 472 with opinion -0.08525929424911327, resistance 0.4655032759544142, influence 1.2143201410493343, and revolt threshold 0.9346087958335353
    Hi, I am Individual 86 with opinion 0.8587992950950754, resistance 0.4433914419594108, influence 0.8489956193014473, and revolt threshold 0.9285976620960174
    Hi, I am Individual 556 with opinion -0.6438060403193133, resistance 0.46603123255063017, influence 0.9427714310309658, and revolt threshold 0.9535662424948872
    Hi, I am Individual 430 with opinion 0.28158944930254237, resistance 0.6742930372806953, influence 1.342573153688442, and revolt threshold 0.712004986651464
    Hi, I am Individual 449 with opinion -0.23671656575958755, resistance 0.6421784407315168, influence 1.2786071931462082, and revolt threshold 0.9206304680565329
    Hi, I am Individual 958 with opinion 0.9309229473148839, resistance 0.9878315224413554, influence 0.8957502925921597, and revolt threshold 0.8690871245429324
    Hi, I am Individual 325 with opinion -0.03134397252187937, resistance 0.7696162237467914, influence 0.9978932557091265, and revolt threshold 0.8255826133686517
    Hi, I am Individual 677 with opinion 0.34436202136488303, resistance 0.8481227852768976, influence 1.3640530439174232, and revolt threshold 0.9508555086466217
    Hi, I am Individual 526 with opinion -0.6785684335845474, resistance 0.8575677721219023, influence 1.3788236063287846, and revolt threshold 0.9939537300670882
    Hi, I am Individual 688 with opinion -0.928133772834957, resistance 0.046838177533830905, influence 1.3147547465722922, and revolt threshold 0.8202808187517575
    Hi, I am Individual 884 with opinion 0.8798408639382056, resistance 0.9828690547457472, influence 1.1180781664869237, and revolt threshold 0.8492372275316807
    Hi, I am Individual 839 with opinion -0.11003616981164432, resistance 0.6860690761043176, influence 1.4773193603822343, and revolt threshold 0.7094022511573292
    Hi, I am Individual 466 with opinion -0.5853746597540281, resistance 0.4008620784870699, influence 1.3082890285316329, and revolt threshold 0.8615605110411196
    Hi, I am Individual 957 with opinion 0.9428432349213405, resistance 0.23264128590794886, influence 1.0461186095036519, and revolt threshold 0.837303929782356
    Hi, I am Individual 477 with opinion 0.7525245583906481, resistance 0.32245346901445704, influence 1.3551961646545974, and revolt threshold 0.9291945093834912
    Hi, I am Individual 330 with opinion 0.9879460545215348, resistance 0.12170035358400577, influence 1.4488691454620006, and revolt threshold 0.7055326847054947
    Hi, I am Individual 134 with opinion 0.7155716248747277, resistance 0.23176471977639412, influence 1.0305277419278278, and revolt threshold 0.88251806304644
    Hi, I am Individual 30 with opinion -0.4945634575101532, resistance 0.45682598925150575, influence 1.3362664662647292, and revolt threshold 0.964840299341138
    Hi, I am Individual 490 with opinion -0.5930378798613742, resistance 0.4533497465494507, influence 1.4863820418375222, and revolt threshold 0.856353492382789
    Hi, I am Individual 857 with opinion -0.9516697387995807, resistance 0.4115056073162229, influence 1.2018963067600992, and revolt threshold 0.9993928083242223
    Hi, I am Individual 975 with opinion 0.5607673483325013, resistance 0.9476520970773816, influence 0.9802713543169044, and revolt threshold 0.8694303234077242
    Hi, I am Individual 352 with opinion -0.16958160461765814, resistance 0.9313874009402885, influence 1.4474465960195024, and revolt threshold 0.903110179681416
    Hi, I am Individual 422 with opinion -0.9031193592243207, resistance 0.9839362726423705, influence 1.0528831375787364, and revolt threshold 0.8503506816096973
    Hi, I am Individual 579 with opinion 0.4478457058712235, resistance 0.3449642224125834, influence 1.468607753383588, and revolt threshold 0.952279564527666
    Hi, I am Individual 913 with opinion 0.23422923946504426, resistance 0.535745317374326, influence 1.1961752089592521, and revolt threshold 0.825671952152416
    Hi, I am Individual 672 with opinion 0.7512676677506798, resistance 0.2559819082269442, influence 1.2804782035253626, and revolt threshold 0.989851192686291
    Hi, I am Individual 652 with opinion 0.6888324768516725, resistance 0.8989822844630732, influence 1.4265664410763674, and revolt threshold 0.9368959561527572
    Hi, I am Individual 335 with opinion -0.43469387809315685, resistance 0.3578785915587267, influence 1.3620326654050656, and revolt threshold 0.8824229766961711
    Hi, I am Individual 658 with opinion 0.23970049417213457, resistance 0.8618438282378384, influence 1.0901514683889424, and revolt threshold 0.7389659041310541
    Hi, I am Individual 781 with opinion 0.578369142838774, resistance 0.42585336082015357, influence 1.2388906950025973, and revolt threshold 0.7493409006661306
    Hi, I am Individual 24 with opinion 0.45113711182658944, resistance 0.8411572343851146, influence 1.1309708439923551, and revolt threshold 0.9846382439876957
    Hi, I am Individual 166 with opinion -0.8922547691705078, resistance 0.09705991204579989, influence 1.2768302037888584, and revolt threshold 0.8218781360993959
    Hi, I am Individual 382 with opinion 0.7621575997334076, resistance 0.5984342539558481, influence 0.8802156152859656, and revolt threshold 0.905530206506114
    Hi, I am Individual 183 with opinion 0.7529279422479827, resistance 0.8255618953410303, influence 0.9433544560707139, and revolt threshold 0.8796100703385887
    Hi, I am Individual 227 with opinion 0.8193770060062473, resistance 0.4956293979540619, influence 1.2178752998883544, and revolt threshold 0.7400174339552849
    Hi, I am Individual 831 with opinion 0.9524109173875206, resistance 0.31220285019113714, influence 1.2886854778146268, and revolt threshold 0.7516656953854063
    Hi, I am Individual 651 with opinion 0.5586353771847339, resistance 0.8744341762290626, influence 0.8771240353742041, and revolt threshold 0.9254832962257223
    Hi, I am Individual 698 with opinion -0.04441848774766699, resistance 0.48143781564746324, influence 1.3871927603625935, and revolt threshold 0.8923784178005617
    Hi, I am Individual 474 with opinion -0.2926976993196262, resistance 0.023389990214404466, influence 1.1596237426325615, and revolt threshold 0.7803662016087614
    Hi, I am Individual 417 with opinion 0.40603751239169794, resistance 0.8760169938322457, influence 0.8970059502643907, and revolt threshold 0.9639722987471152
    Hi, I am Individual 480 with opinion -0.6206780802802594, resistance 0.08383484703324318, influence 1.3297564348306796, and revolt threshold 0.9424292683153508
    Hi, I am an propaganda agent, you can call me Propaganda Hub 1032 and I have the id of 1032.       I have the opinion of -1, and influence strength is 0.7529433544909598      with a radius of 68.5531830761344.
    Hi, I am Individual 953 with opinion -0.8696107770123731, resistance 0.5168530987416228, influence 0.9833207612757366, and revolt threshold 0.8993181000293771
    Hi, I am Individual 879 with opinion 0.5142900088535478, resistance 0.8957819607478005, influence 1.2597271969158856, and revolt threshold 0.9878379724857962
    Hi, I am Individual 52 with opinion -0.20925791241790903, resistance 0.237877651162005, influence 1.3683592684381916, and revolt threshold 0.7693902409740023
    Hi, I am Individual 267 with opinion 0.6410021562108008, resistance 0.714587552993704, influence 1.3181753086281822, and revolt threshold 0.9055611197555056
    Hi, I am Individual 228 with opinion -0.4353131696587702, resistance 0.5327415595432556, influence 0.8700364556904757, and revolt threshold 0.8568032345894308
    Hi, I am Individual 225 with opinion -0.8907648307378417, resistance 0.7285743085128676, influence 1.2483645108812362, and revolt threshold 0.879327442890649
    Hi, I am Individual 274 with opinion 0.0026698029319112138, resistance 0.521049635632579, influence 1.4391397345809573, and revolt threshold 0.8066599609992876
    Hi, I am Individual 118 with opinion -0.059571137964196774, resistance 0.9040521817530411, influence 1.1710647895315178, and revolt threshold 0.7600989273695538
    Hi, I am Individual 942 with opinion 0.9690875037573987, resistance 0.9988252653955779, influence 1.2839759665247708, and revolt threshold 0.7291876357788296
    Hi, I am Individual 94 with opinion -0.8706945773633341, resistance 0.9268076052711225, influence 1.2197028224239324, and revolt threshold 0.9831645961252506
    Hi, I am Individual 560 with opinion 0.26091501071224843, resistance 0.26991492224753943, influence 1.1851858167445077, and revolt threshold 0.8194563797210208
    Hi, I am Individual 559 with opinion 0.6405604359405039, resistance 0.8312529734700869, influence 1.4195807750450735, and revolt threshold 0.7647796193232865
    Hi, I am Individual 461 with opinion 0.4059658609955046, resistance 0.4888385315020102, influence 1.0740492938505892, and revolt threshold 0.8515308250423794
    Hi, I am Individual 491 with opinion 0.34766645254749173, resistance 0.9049852750523389, influence 1.285999627941281, and revolt threshold 0.7749401864528649
    Hi, I am Individual 264 with opinion -0.3567042576022055, resistance 0.15026688080736073, influence 1.0608207013177362, and revolt threshold 0.8954537444836709
    Hi, I am Individual 153 with opinion -0.4877511843792868, resistance 0.08995554870444089, influence 1.430104308951611, and revolt threshold 0.8301365219094047
    Hi, I am Individual 404 with opinion 0.35422382866325464, resistance 0.254519232664714, influence 1.2860287115976337, and revolt threshold 0.8667136855635585
    Hi, I am Individual 469 with opinion -0.5401126093635777, resistance 0.20538081785581197, influence 0.814567319226936, and revolt threshold 0.9725042741569432
    Hi, I am Individual 800 with opinion -0.6025404880756644, resistance 0.5013259794867907, influence 1.0641568576096563, and revolt threshold 0.9583582004885345
    Hi, I am Individual 137 with opinion -0.1373565921722295, resistance 0.19953250194160776, influence 1.2443892141951232, and revolt threshold 0.9073549363917448
    Hi, I am Individual 827 with opinion -0.7928314561544554, resistance 0.4846973889919973, influence 0.8746793026869192, and revolt threshold 0.9690386532874877
    Hi, I am Individual 799 with opinion 0.7624294967362284, resistance 0.26541265207687115, influence 1.2088698621985832, and revolt threshold 0.8877514400798754
    Hi, I am Individual 740 with opinion 0.2975044789781993, resistance 0.9238429441233476, influence 1.304504966535072, and revolt threshold 0.8384105456838798
    Hi, I am Individual 316 with opinion 0.9986662426069006, resistance 0.30536376618618366, influence 1.3122750158020766, and revolt threshold 0.8007022655993397
    Hi, I am an propaganda agent, you can call me Propaganda Hub 1046 and I have the id of 1046.       I have the opinion of -1, and influence strength is 0.9956801007391755      with a radius of 56.17444304424481.
    Hi, I am Individual 860 with opinion -0.544349951532273, resistance 0.6985173774289933, influence 1.0115288466292762, and revolt threshold 0.782463512693014
    Hi, I am an propaganda agent, you can call me Propaganda Hub 1003 and I have the id of 1003.       I have the opinion of -1, and influence strength is 0.9202541945737752      with a radius of 45.22949260356131.
    Hi, I am Individual 372 with opinion -0.993781008171795, resistance 0.1950700571787113, influence 1.3555197375130432, and revolt threshold 0.9295689843120438
    Hi, I am Individual 732 with opinion -0.8698984845953097, resistance 0.6870898381545578, influence 1.075630263761869, and revolt threshold 0.9754937894644594
    Hi, I am Individual 258 with opinion -0.6537698238662493, resistance 0.33533088245662945, influence 1.013114684833026, and revolt threshold 0.7178806861808662
    Hi, I am Individual 223 with opinion 0.029927038292521413, resistance 0.6092398546461804, influence 1.0832676752837143, and revolt threshold 0.9450588835543577
    Hi, I am Individual 321 with opinion 0.45131692773698795, resistance 0.4782362461615567, influence 1.035208323446316, and revolt threshold 0.8969471366670323
    Hi, I am Individual 576 with opinion 0.7292449710344022, resistance 0.22577140392834216, influence 1.232810458673602, and revolt threshold 0.7819812367474223
    Hi, I am Individual 737 with opinion -0.8109204767353242, resistance 0.9699813892993279, influence 1.3019384600766073, and revolt threshold 0.7356178903185808
    Hi, I am Individual 936 with opinion 0.5930038846942316, resistance 0.06697706453934005, influence 0.8693979378184821, and revolt threshold 0.8170215435708676
    Hi, I am Individual 820 with opinion 0.897139791214236, resistance 0.9820168099391824, influence 1.4188159632283739, and revolt threshold 0.9990264742776127
    Hi, I am an propaganda agent, you can call me Propaganda Hub 1043 and I have the id of 1043.       I have the opinion of 1, and influence strength is 0.8948944693326917      with a radius of 79.37607409342499.
    Hi, I am Individual 413 with opinion -0.9225634385522994, resistance 0.86663874547496, influence 1.3916408616864135, and revolt threshold 0.8546443938300371
    Hi, I am Individual 819 with opinion -0.2595347242718695, resistance 0.7245309339688522, influence 1.2456901311564734, and revolt threshold 0.757870113596281
    Hi, I am Individual 339 with opinion 0.3907135245222222, resistance 0.5641824326125247, influence 0.9708341989616684, and revolt threshold 0.7976040302819476
    Hi, I am an propaganda agent, you can call me Propaganda Hub 1017 and I have the id of 1017.       I have the opinion of 1, and influence strength is 0.5719456094011381      with a radius of 60.65700955805244.
    Hi, I am Individual 222 with opinion -0.08075136354122514, resistance 0.031243790111174508, influence 0.9170674969510774, and revolt threshold 0.7028407315803037
    Hi, I am Individual 401 with opinion -0.28299331291856555, resistance 0.7115646212948147, influence 1.1358503707806142, and revolt threshold 0.8582021017807749
    Hi, I am Individual 926 with opinion -0.9523962978301781, resistance 0.6757378655178997, influence 1.2575003358170416, and revolt threshold 0.7602745056291621
    Hi, I am Individual 917 with opinion -0.934756547754283, resistance 0.44815945654829314, influence 1.3373289942976028, and revolt threshold 0.8395390738288718
    Hi, I am Individual 456 with opinion 0.08452065147908461, resistance 0.9625339157771089, influence 0.8831894282269412, and revolt threshold 0.9525854278709291
    Hi, I am Individual 250 with opinion 0.13606391755381475, resistance 0.4050109327416169, influence 1.0223500152028353, and revolt threshold 0.7956643941277326
    Hi, I am Individual 899 with opinion -0.36780473425125426, resistance 0.2874992813950119, influence 0.8543210030767165, and revolt threshold 0.7004318504257064
    Hi, I am Individual 645 with opinion -0.702854947726413, resistance 0.2032887721487341, influence 0.8707277950400549, and revolt threshold 0.8901834843879302
    Hi, I am Individual 31 with opinion 0.5849169206337834, resistance 0.4812579585325627, influence 1.366639793940303, and revolt threshold 0.8203529285641366
    Hi, I am an propaganda agent, you can call me Propaganda Hub 1011 and I have the id of 1011.       I have the opinion of 1, and influence strength is 0.5938180039709948      with a radius of 99.69331782413911.
    Hi, I am Individual 587 with opinion 0.135904093301221, resistance 0.7071021804163468, influence 0.8184592892588557, and revolt threshold 0.7143153243819291
    Hi, I am Individual 18 with opinion -0.379551811525386, resistance 0.06691251162121092, influence 1.0077429487586156, and revolt threshold 0.8146804825541745
    Hi, I am Individual 569 with opinion 0.9057676503870502, resistance 0.8474790934502823, influence 1.3098597562095091, and revolt threshold 0.9095068604112773
    Hi, I am Individual 205 with opinion -0.45604406351573723, resistance 0.5987996803859723, influence 1.1452979304003157, and revolt threshold 0.8847942768774624
    Hi, I am Individual 150 with opinion 0.7167480014167746, resistance 0.5996964094173559, influence 1.4423546236693716, and revolt threshold 0.835248728778071
    Hi, I am Individual 964 with opinion 0.03868258768103905, resistance 0.7197498746933898, influence 1.1903996093073106, and revolt threshold 0.7548865035634481
    Hi, I am Individual 216 with opinion 0.2677800989346555, resistance 0.4292481702830804, influence 0.8867536267345227, and revolt threshold 0.8484925389218143
    Hi, I am Individual 710 with opinion 0.7125577148356699, resistance 0.4100316393673351, influence 1.075652150788552, and revolt threshold 0.8437763600813981
    Hi, I am Individual 809 with opinion 0.5618087232742728, resistance 0.3245134357919143, influence 1.018498431929976, and revolt threshold 0.7607825737925399
    Hi, I am Individual 171 with opinion -0.06867510122325404, resistance 0.7083266329006643, influence 1.1004736894763187, and revolt threshold 0.8971812315147788
    Hi, I am Individual 47 with opinion -0.4564520866400026, resistance 0.20292523111722216, influence 1.231382919689494, and revolt threshold 0.8600185351131947
    Hi, I am Individual 204 with opinion -0.0707149671720615, resistance 0.023977162305850896, influence 0.8671549892561657, and revolt threshold 0.9253301059262156
    Hi, I am Individual 858 with opinion 0.022461539805919806, resistance 0.41000100516225846, influence 1.1234592771905647, and revolt threshold 0.7614347856457339
    Hi, I am Individual 684 with opinion 0.44260474228881574, resistance 0.2925284751500197, influence 1.3241656837456166, and revolt threshold 0.7487987684262286
    Hi, I am Individual 429 with opinion -0.39156720755662144, resistance 0.33423533724616417, influence 1.4210443885176893, and revolt threshold 0.7423084636065304
    Hi, I am Individual 177 with opinion 0.32046116270135117, resistance 0.0012651976716089308, influence 1.3147521571356486, and revolt threshold 0.9245486199981918
    Hi, I am Individual 300 with opinion -0.502572786122963, resistance 0.622933730555347, influence 0.9272489141430629, and revolt threshold 0.8152082117554134
    Hi, I am Individual 272 with opinion 0.14707612070666798, resistance 0.23477838633389836, influence 1.4326821861295111, and revolt threshold 0.7572327485828558
    Hi, I am Individual 237 with opinion 0.18742088948581803, resistance 0.5025890045650272, influence 0.8301612658587408, and revolt threshold 0.7131907002418519
    Hi, I am Individual 61 with opinion -0.29411392302132455, resistance 0.40857747506449016, influence 0.8121678697224017, and revolt threshold 0.8554028313874785
    Hi, I am Individual 27 with opinion 0.9746170109137509, resistance 0.6727163495229427, influence 1.135851262239091, and revolt threshold 0.8351303054583061
    Hi, I am Individual 147 with opinion 0.04060127006511971, resistance 0.5957528717830205, influence 0.9384894758068638, and revolt threshold 0.9017872062419161
    Hi, I am Individual 657 with opinion -0.7066407951820848, resistance 0.6036347143190045, influence 1.0156189978989743, and revolt threshold 0.8068561238296619
    Hi, I am Individual 607 with opinion 0.6148750452573533, resistance 0.13491377168637708, influence 1.1701640915672398, and revolt threshold 0.9818770525183635
    Hi, I am Individual 622 with opinion 0.3975083159047619, resistance 0.2600996939938517, influence 1.0156013840327618, and revolt threshold 0.8222714117878468
    Hi, I am Individual 510 with opinion -0.9943121070134882, resistance 0.47335416060217206, influence 0.8964778787857307, and revolt threshold 0.7815086916065213
    Hi, I am Individual 122 with opinion 0.5680411639772112, resistance 0.22719895153543412, influence 0.8455338237916987, and revolt threshold 0.7901101601422682
    Hi, I am Individual 313 with opinion 0.6819995513546393, resistance 0.8802707306913192, influence 0.850081943008056, and revolt threshold 0.8896600214812589
    Hi, I am Individual 976 with opinion -0.4148082633834851, resistance 0.7509472832458512, influence 0.8981773327020095, and revolt threshold 0.7915291462195002
    Hi, I am Individual 726 with opinion 0.626850668527474, resistance 0.9436338153495689, influence 1.10552983425509, and revolt threshold 0.8990133559480965
    Hi, I am Individual 867 with opinion 0.636224768048719, resistance 0.8974514109984092, influence 1.0329570083093729, and revolt threshold 0.842992802302213
    Hi, I am Individual 304 with opinion 0.3770854891798405, resistance 0.5465327957789028, influence 1.1820835878273206, and revolt threshold 0.707365144956991
    Hi, I am Individual 196 with opinion 0.21510558350636266, resistance 0.7846148377019124, influence 1.049719933730224, and revolt threshold 0.8049143038154839
    Hi, I am Individual 903 with opinion -0.0860546643153477, resistance 0.428489839478821, influence 1.4628865281489092, and revolt threshold 0.7072402843458572
    Hi, I am Individual 841 with opinion 0.2926792216171399, resistance 0.32695384998555455, influence 1.3598479793156675, and revolt threshold 0.7301432589129617
    Hi, I am Individual 551 with opinion 0.09074186581841581, resistance 0.779096539067554, influence 1.2675700677549915, and revolt threshold 0.7821380154074242
    Hi, I am Individual 584 with opinion -0.39824737023876744, resistance 0.44976339837118706, influence 1.224703041050871, and revolt threshold 0.8304208230341406
    Hi, I am Individual 487 with opinion 0.9959311125147394, resistance 0.6153940945793844, influence 1.303609368670445, and revolt threshold 0.9802647104953721
    Hi, I am Individual 180 with opinion 0.9831573601549117, resistance 0.36654909433283156, influence 1.141954424792751, and revolt threshold 0.7834510742012494
    Hi, I am Individual 612 with opinion 0.27757208914964515, resistance 0.5494460940286178, influence 1.252638327144261, and revolt threshold 0.7282348722363305
    Hi, I am Individual 756 with opinion 0.4834792455064847, resistance 0.45060005136109826, influence 0.8378104713712414, and revolt threshold 0.9564743036322131
    Hi, I am Individual 212 with opinion 0.5270400151497976, resistance 0.5984644321672545, influence 0.8793032077823475, and revolt threshold 0.9823551310731365
    Hi, I am Individual 41 with opinion -0.528677983650949, resistance 0.20636439603019818, influence 1.4409981339320912, and revolt threshold 0.7785995301832228
    Hi, I am Individual 347 with opinion -0.6519297608541756, resistance 0.9654226280759027, influence 1.309663915594986, and revolt threshold 0.7084229331441773
    Hi, I am Individual 760 with opinion 0.6354845774112245, resistance 0.5607861968770009, influence 1.0566971172339552, and revolt threshold 0.7300333243818481
    Hi, I am Individual 12 with opinion 0.6798844519919527, resistance 0.5210076302111882, influence 1.3496293091536697, and revolt threshold 0.7808788137618291
    Hi, I am Individual 916 with opinion 0.9024659298652458, resistance 0.635200737307851, influence 1.3539760402978953, and revolt threshold 0.9529066195129785
    Hi, I am Individual 365 with opinion -0.27635691339790847, resistance 0.5440927127207411, influence 1.4923295695028782, and revolt threshold 0.8365659322873509
    Hi, I am Individual 39 with opinion 0.34491173407690967, resistance 0.9477960222505795, influence 1.103636901290189, and revolt threshold 0.9715880720237511
    Hi, I am Individual 5 with opinion 0.8353064770353871, resistance 0.09101591576673806, influence 1.4831019061882076, and revolt threshold 0.9335997201845955
    Hi, I am Individual 85 with opinion 0.9656401003553221, resistance 0.5180929634191535, influence 1.4672743967331057, and revolt threshold 0.9867911762754061
    Hi, I am Individual 996 with opinion -0.7915024523789014, resistance 0.304637345375946, influence 0.8705038892897146, and revolt threshold 0.8574002676691593
    Hi, I am an propaganda agent, you can call me Propaganda Hub 1048 and I have the id of 1048.       I have the opinion of 1, and influence strength is 0.7274347856250114      with a radius of 62.212975698890816.
    Hi, I am Individual 229 with opinion 0.23666099093036608, resistance 0.13767657626452867, influence 1.0173250736991286, and revolt threshold 0.7687737744571326
    Hi, I am Individual 36 with opinion -0.20646273471236043, resistance 0.13525567838629982, influence 1.0479538721284873, and revolt threshold 0.9386166186023456
    Hi, I am Individual 937 with opinion 0.7174131336091674, resistance 0.18911298450376435, influence 1.3177645082452818, and revolt threshold 0.9526968544897871
    Hi, I am Individual 282 with opinion -0.6125229732115465, resistance 0.6174881756002533, influence 0.9108879372195798, and revolt threshold 0.9634434523718588
    Hi, I am Individual 380 with opinion -0.33472681335854015, resistance 0.5642280486826292, influence 1.2490585611676286, and revolt threshold 0.9456940957241025
    Hi, I am Individual 680 with opinion 0.4992499336788725, resistance 0.7027353686823623, influence 1.1472734442690469, and revolt threshold 0.9650631031995639
    Hi, I am Individual 705 with opinion 0.0638395797858251, resistance 0.2652661925065225, influence 1.1724285052259242, and revolt threshold 0.7002982653592257
    Hi, I am Individual 730 with opinion -0.2338540972683738, resistance 0.5620039710046532, influence 0.8106792357380516, and revolt threshold 0.8454093434188158
    Hi, I am Individual 164 with opinion 0.6228662405903866, resistance 0.3063118111515021, influence 0.91327033372777, and revolt threshold 0.7011584823329884
    Hi, I am Individual 624 with opinion -0.1386049913454792, resistance 0.32225056424076604, influence 1.050330169707193, and revolt threshold 0.9171360187485053
    Hi, I am Individual 285 with opinion -0.6133184985534958, resistance 0.9988901293257891, influence 0.8419491977119911, and revolt threshold 0.799809012752421
    Hi, I am Individual 602 with opinion 0.11618099406510285, resistance 0.10568624709975216, influence 1.4666291625522634, and revolt threshold 0.8221341822880374
    Hi, I am Individual 460 with opinion 0.5787401991520829, resistance 0.3546699892810772, influence 1.3866223637095736, and revolt threshold 0.9526340304436207
    Say hi execution time: 0.005464981004479341 seconds


    /tmp/ipykernel_1047370/4281277965.py:6: DeprecationWarning: The time module and all its Schedulers are deprecated and will be removed in Mesa 3.1. They can be replaced with AgentSet functionality. See the migration guide for details. https://mesa.readthedocs.io/latest/migration_guide.html#time-and-schedulers
      self.schedule = mesa.time.RandomActivation(self)



```python
step_execution_time = timeit.timeit(run_abm_step, number=15)
print(f"ABM init time: {ABM_init_time} seconds")
print(f"Say hi execution time: {sayhi_execution_time} seconds")
print(f"ABM step execution time: {step_execution_time} seconds")
```


    Current step is 1!
    Current step is 2!
    Current step is 3!
    Current step is 4!
    Current step is 5!
    Current step is 6!
    Current step is 7!
    Current step is 8!
    Current step is 9!
    Current step is 10!
    Current step is 11!
    Current step is 12!
    Current step is 13!
    Current step is 14!
    Current step is 15!
    ABM init time: 0.14338544900238048 seconds
    Say hi execution time: 0.005464981004479341 seconds
    ABM step execution time: 565.5710444339929 seconds



```python
# visualize_data(starter_model)
```


```python
starter_model.visualize_metrics()
```


    
![png](output_10_0.png)
    



    
![png](output_10_1.png)
    



    
![png](output_10_2.png)
    



    
![png](output_10_3.png)
    



    
![png](output_10_4.png)
    

