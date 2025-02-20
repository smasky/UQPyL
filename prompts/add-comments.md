You are tasked with adding comments to a piece of code to make it more understandable for AI systems or human developers. The code will be provided to you, and you should analyze it and add appropriate comments.

The most and most important things are to read and learn carefully *.py files, the original comments are not all right.

To add comments to this code, follow these steps:

1. Analyze the code to understand its structure and functionality.
2. Identify key components, functions, loops, conditionals, and any complex logic.
3. Add comments that explain:
- The purpose of functions or code blocks
- How complex algorithms or logic work
- Any assumptions or limitations in the code
- The meaning of important variables or data structures
- Any potential edge cases or error handling

When adding comments, follow these guidelines:

- Use clear and concise language
- Avoid stating the obvious (e.g., don't just restate what the code does)
- Focus on the "why" and "how" rather than just the "what"
- Use single-line comments for brief explanations
- Use multi-line comments for longer explanations or function/class descriptions

Following content is example (Class and Function) for comments:
Class:
```
class Delta_Test(SA):
    """
    -------------------------------------------------
    Delta Test
    -------------------------------------------------
    This class implements the Delta Test, which is 
    a non-parametric method for sensitivity analysis.
    
    Methods:
        sample: Generate a sample for Delta Test analysis
        analyze: Perform Delta Test analysis from the X and Y you provided.
        findCombEA: Find the best combination using Evolutionary Algorithm.
        findCombVio: Find the best combination using brute-force approach.
    
    Examples:
        # `problem` is an instance of ProblemABC or Problem from UQPyL.problems
        # You must create a problem instance before using this method.
        >>> delta_method = Delta_Test(nNeighbors=2)
        >>> X = delta_method.sample(problem, N=1000)
        >>> res = delta_method.analyze(problem, X)
        >>> print(res)
        
    References:
        [1] E. Eirola et al, Using the Delta Test for Variable Selection, 
            Artificial Neural Networks, 2008.
        [2] SALib, https://github.com/SALib/SALib
    -------------------------------------------------
    """
```
Function:
```
    def sample(self, problem: Problem, N: int = 500, sampler: Sampler = LHS('classic')):
        """
        Generate a sample set for the Delta Test.
        --------------------------------------------------
        :param problem: Problem - The problem instance defining the input space.
        :param N: int - The number of samples to generate. Defaults to 500.
        :param sampler: Sampler - The sampling method to use. Defaults to Latin Hypercube Sampling (LHS) with 'classic' mode.

        :return: np.ndarray - A 2D array of shape `(N, nInput)`, where `nInput` is the number of input variables.
        
        
        nInput = problem.nInput
        
        # Generate samples using the specified sampler
        X = sampler.sample(N, nInput)
        
        # Transform the samples to the problem's input space
        return problem._transform_unit_X(X)
        """
```

Your output should be the original code with your added comments. Make sure to preserve the original code's formatting and structure.

Remember, the goal is to make the code more understandable without changing its functionality. Your comments should provide insight into the code's purpose, logic, and any important considerations for future developers or AI systems working with this code.

When you add comments for *.py files, please use complete english language.

