# Importing all the necessary libraries
import sys
import time
import os
import shutil
import random
import importlib
import numpy as np
import matplotlib.pyplot as plt
import KratosMultiphysics

# Create a new folder to store all result folders
kratos_results_folder = "Kratos_Results"
if not os.path.exists(kratos_results_folder):
    os.makedirs(kratos_results_folder)

# Create a new folder to store junk files
junk_folder = os.path.join(kratos_results_folder, "junk")
if not os.path.exists(junk_folder):
    os.makedirs(junk_folder)

# Modify the delete_result_folders function to delete folders inside Kratos_Results
def delete_result_folders():
    """
    Deletes the folders 'displacement_results', 'loading_results', 'stiffness_results', and 'mass_results' 
    from the 'Kratos_Results' directory.
    
    This ensures a clean working environment by removing previous simulation result folders.
    """
    cwd = os.getcwd()  # Get the current working directory
    kratos_results_path = cwd #os.path.join(cwd, kratos_results_folder)
    folders_to_delete = ["Kratos_Results"]  # List of folder names to delete
    
    for folder in folders_to_delete:
        folder_path = os.path.join(kratos_results_path, folder)  # Construct the full path to the folder
        if os.path.exists(folder_path):  # Check if the folder existsA
            shutil.rmtree(folder_path)  # Delete the folder and its contents
            print(f"Deleted folder: {folder_path}")
        else:
            print(f"Folder not found: {folder_path}")

# Call the function to delete existing .npy files
delete_result_folders()

# Generate Random list of 'mu' values for parameterization
def generate_random_list_of_mu(num_sets=5, min_value=-300.0, max_value=400.0, size=3):
    """
    Generates a list of modulus value sets with two decimal precision.
    
    Args:
        num_sets (int): Number of sets to generate.
        min_value (float): Minimum value for random generation.
        max_value (float): Maximum value for random generation.
        size (int): Number of values in each set.
    
    Returns:
        list: A list containing `num_sets` sets of random modulus values.
    """
    random_list_of_mu = []
    for _ in range(num_sets):
        # Generate a random set of 'size' values within the given range
        random_set = [round(random.uniform(min_value, max_value), 2) for _ in range(size)]
        random_list_of_mu.append(random_set)
    return random_list_of_mu

# Generate the list_of_mu with random values
list_of_mu = generate_random_list_of_mu()


# Assigning the values of 'mu' for each parametric iteration
def UpdateProjectParameters(parameters, mu):
    """
    Updates the project parameters with specified modulus values.
    
    Args:
        parameters (dict): The simulation parameters.
        mu (list): A list of modulus values to assign.
    
    Returns:
        dict: Updated parameters with new modulus values.
    """
    # Update the modulus values for each load in the loads_process_list
    for i in range(len(mu)):
        if parameters["processes"]["loads_process_list"][i]["Parameters"].Has("value"):
            parameters["processes"]["loads_process_list"][i]["Parameters"]["value"].SetDouble(mu[i])
        elif parameters["processes"]["loads_process_list"][i]["Parameters"].Has("modulus"):
            parameters["processes"]["loads_process_list"][i]["Parameters"]["modulus"].SetDouble(mu[i])
    return parameters


# Main pre-existing Kratos function with necessary alterations
def CreateAnalysisStageWithFlushInstance(cls, global_model, parameters):
    """
    Creates a custom analysis stage class with functionality for periodic flushing
    and saving nodal displacements and global force vectors.
    
    Args:
        cls: Base class of the analysis stage.
        global_model (KratosMultiphysics.Model): The Kratos model.
        parameters (KratosMultiphysics.Parameters): Simulation parameters.
    
    Returns:
        object: An instance of the customized analysis stage.
    """
    class AnalysisStageWithFlush(cls):
        def __init__(self, model, project_parameters, flush_frequency=10.0, modulus_values=None):
            """
            Initializes the analysis stage with flushing capabilities.
            
            Args:
                model: The Kratos model.
                project_parameters: Parameters for the analysis.
                flush_frequency (float): Frequency to flush output.
                modulus_values (list): Modulus values for the current iteration.
            """
            super().__init__(model, project_parameters)
            self.flush_frequency = flush_frequency
            self.last_flush = time.time()
            self.nodal_displacements = []
            self.modulus_values = modulus_values  # Store the modulus values
            sys.stdout.flush()

        def Initialize(self):
            """
            Initializes the simulation by calling the base Initialize method
            and flushing the output.
            """
            super().Initialize()
            sys.stdout.flush()

        def FinalizeSolutionStep(self):
            """
            Finalizes the solution step, collecting nodal displacements, force vector, stiffness matrix,
            and mass matrix.
            """
            super().FinalizeSolutionStep()

            # Collect nodal displacements for the current solution step
            nodal_displacements_current_time_step = []
            model_part = self.model.GetModelPart("Structure")

            for node in model_part.Nodes:
                # Append displacement values for each node
                nodal_displacements_current_time_step.append(node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT))

            self.nodal_displacements.append(nodal_displacements_current_time_step)

            # Assemble global force vector, stiffness matrix, and mass matrix
            f_global = self.AssembleGlobalForceVector(model_part)
            K_global = self.AssembleGlobalStiffnessMatrix(model_part)
            M_global = self.AssembleGlobalLumpedMassMatrix(model_part)

            # Save the global force vector to a file
            output_folder = os.path.join(kratos_results_folder, "loading_results")
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            force_vector_file_path = os.path.join(output_folder, f"global_force_vector_{self.modulus_values}.npy")
            np.save(force_vector_file_path, f_global)
            print(f"Force vector saved to: {force_vector_file_path}")

            # Save the global stiffness matrix to a file
            stiff_output_folder = os.path.join(kratos_results_folder, "stiffness_results")
            if not os.path.exists(stiff_output_folder):
                os.makedirs(stiff_output_folder)
            stiffness_matrix_file_path = os.path.join(stiff_output_folder, f"stiffness_matrix_{self.modulus_values}.npy")
            np.save(stiffness_matrix_file_path, K_global)
            print(f"Stiffness matrix saved to: {stiffness_matrix_file_path}")

            # Save the global mass matrix to a file
            mass_output_folder = os.path.join(kratos_results_folder, "mass_results")
            if not os.path.exists(mass_output_folder):
                os.makedirs(mass_output_folder)
            mass_matrix_file_path = os.path.join(mass_output_folder, f"mass_matrix_{self.modulus_values}.npy")
            np.save(mass_matrix_file_path, M_global)
            print(f"Mass matrix saved to: {mass_matrix_file_path}")

            # Flush the output periodically
            if self.parallel_type == "OpenMP":
                now = time.time()
                if now - self.last_flush > self.flush_frequency:
                    sys.stdout.flush()
                    self.last_flush = now
        
        def AssembleGlobalForceVector(self, model_part):
            """
            Assembles the global force vector from local contributions.
            
            Args:
                model_part: The computational model part.
            
            Returns:
                numpy.ndarray: The assembled global force vector.
            """
            builder_and_solver = self._GetSolver()._GetBuilderAndSolver()
            num_equations = builder_and_solver.GetEquationSystemSize()
            f_global = np.zeros(num_equations)

            # Assemble contributions from conditions
            for condition in model_part.Conditions:
                rhs_cond = KratosMultiphysics.Vector()
                condition.CalculateRightHandSide(rhs_cond, model_part.ProcessInfo)
                equation_ids_cond = condition.EquationIdVector(model_part.ProcessInfo)
                for i in range(len(equation_ids_cond)):
                    eq_id = equation_ids_cond[i]
                    f_global[eq_id] += rhs_cond[i]

            # Assemble contributions from elements
            for element in model_part.Elements:
                rhs_elem = KratosMultiphysics.Vector()
                element.CalculateRightHandSide(rhs_elem, model_part.ProcessInfo)
                equation_ids_elem = element.EquationIdVector(model_part.ProcessInfo)
                for i in range(len(equation_ids_elem)):
                    eq_id = equation_ids_elem[i]
                    f_global[eq_id] += rhs_elem[i]

            return f_global
        
        def AssembleGlobalStiffnessMatrix(self, model_part):
            """
            Assembles the global stiffness matrix by iterating over elements.

            Args:
                model_part (KratosMultiphysics.ModelPart): The computational model part.

            Returns:
                np.ndarray: The assembled global stiffness matrix.
            """
            # Get the builder and solver to determine the size of the global system
            builder_and_solver = self._GetSolver()._GetBuilderAndSolver()
            num_equations = builder_and_solver.GetEquationSystemSize()
            K_global = np.zeros((num_equations, num_equations))

            # Get the ProcessInfo from the model part
            process_info = model_part.ProcessInfo

            # Assemble contributions from elements
            for element in model_part.Elements:
                lhs_elem = KratosMultiphysics.Matrix()  # Local stiffness matrix
                rhs_elem = KratosMultiphysics.Vector()  # Right-hand side vector
                equation_ids_elem = element.EquationIdVector(process_info)

                # Calculate the local stiffness matrix
                element.CalculateLocalSystem(lhs_elem, rhs_elem, process_info)

                # Add element contributions to the global stiffness matrix
                for i in range(len(equation_ids_elem)):
                    for j in range(len(equation_ids_elem)):
                        eq_id_i = equation_ids_elem[i]
                        eq_id_j = equation_ids_elem[j]
                        K_global[eq_id_i, eq_id_j] += lhs_elem[i, j]

            return K_global
        
        def AssembleGlobalLumpedMassMatrix(self, model_part):
            """
            Assembles the global lumped mass matrix by iterating over elements.

            Args:
                model_part (KratosMultiphysics.ModelPart): The computational model part.

            Returns:
                np.ndarray: The assembled global lumped mass matrix.
            """
            # Get the builder and solver to determine the size of the global system
            builder_and_solver = self._GetSolver()._GetBuilderAndSolver()
            num_equations = builder_and_solver.GetEquationSystemSize()
            M_global = np.zeros((num_equations, num_equations))

            # Get the ProcessInfo from the model part
            process_info = model_part.ProcessInfo

            # Assemble contributions from elements
            for element in model_part.Elements:
                lhs_elem = KratosMultiphysics.Matrix()  # Local stiffness matrix
                rhs_elem = KratosMultiphysics.Vector()  # Right-hand side vector
                equation_ids_elem = element.EquationIdVector(process_info)

                # Calculate the local mass matrix
                element.CalculateMassMatrix(lhs_elem, process_info)

                # Lump the mass matrix (convert consistent to lumped)
                lumped_mass = np.sum(lhs_elem, axis=1)

                # Add element contributions to the global mass matrix
                for i, eq_id in enumerate(equation_ids_elem):
                    M_global[eq_id, eq_id] += lumped_mass[i]

            return M_global


        def Finalize(self):
            """
            Finalizes the simulation, saving nodal displacements to a file in a new folder.
            """
            super().Finalize()

            # Create a new folder to store the displacement files
            output_folder = os.path.join(kratos_results_folder, "displacement_results")
            if not os.path.exists(output_folder):  # Check if the folder already exists
                os.makedirs(output_folder)  # Create the folder if it doesn't exist
            
            # Convert the displacements list to a NumPy array
            displacements_array = np.array(self.nodal_displacements)
            
            # Save the displacement file in the new folder
            file_path = os.path.join(output_folder, f"displacements_modulus_{self.modulus_values}.npy")
            np.save(file_path, displacements_array)
            print(f"Displacements saved to: {file_path}")
            print(f"Displacements shape: {displacements_array.shape}")

    return AnalysisStageWithFlush(global_model, parameters)


# Main program loop to execute simulations
if __name__ == "__main__":
    for mu in list_of_mu:
        with open("ProjectParameters.json", 'r') as parameter_file:
            parameters = KratosMultiphysics.Parameters(parameter_file.read())

        # Update the modulus values for the current iteration
        parameters = UpdateProjectParameters(parameters, mu)

        # Load the analysis stage class dynamically
        analysis_stage_module_name = parameters["analysis_stage"].GetString()
        analysis_stage_class_name = analysis_stage_module_name.split('.')[-1]
        analysis_stage_class_name = ''.join(x.title() for x in analysis_stage_class_name.split('_'))

        analysis_stage_module = importlib.import_module(analysis_stage_module_name)
        analysis_stage_class = getattr(analysis_stage_module, analysis_stage_class_name)

        # Create and run the simulation instance
        global_model = KratosMultiphysics.Model()
        simulation = CreateAnalysisStageWithFlushInstance(analysis_stage_class, global_model, parameters)

        print(f"Running simulation with load moduli: {mu}")
        simulation.modulus_values = mu
        simulation.Run()
        print(f"Completed simulation for moduli: {mu}")

        # Optional delay to avoid performance issues
        time.sleep(1)
    # Move .lst files and output folders to junk folder after all simulations
    time.sleep(1)  # Give system time to release file handles
    lst_files = [f for f in os.listdir() if f.endswith('.lst')]
    for lst_file in lst_files:
        try:
            dest_path = os.path.join(junk_folder, lst_file)
            if os.path.exists(dest_path):
                os.remove(dest_path)  # Remove if file already exists in destination
            shutil.move(lst_file, dest_path)
            print(f"Moved {lst_file} to junk folder")
        except Exception as e:
            print(f"Error moving {lst_file}: {e}")
            try:
                shutil.copy2(lst_file, os.path.join(junk_folder, lst_file))
                os.remove(lst_file)
                print(f"Successfully copied and removed {lst_file}")
            except Exception as e2:
                print(f"Failed to copy {lst_file}: {e2}")

    output_folders = ["gid_output", "vtk_output"]
    for folder in output_folders:
        if os.path.exists(folder):
            try:
                target_folder = os.path.join(junk_folder, folder)
                shutil.move(folder, target_folder)
                print(f"Moved {folder} to junk folder")
            except Exception as e:
                print(f"Error moving {folder}: {e}")
        else:
            print(f"{folder} does not exist")
    path = os.getcwd()
