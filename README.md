# pyvectors

## About pyvectors

pyvectors is a mathematical module providing (real) 3-dimensional position Vectors centered at the Origin as Python objects through a class called `Vector`.

*Date of creation:* `September 27, 2020` \
*Date of first release on [PyPI](https://pypi.org/):* `April 29, 2021`

Using this module, vector algebra can be performed fairly easily in Python. It allows a user to plot Vectors on a 3D plot, and can be used as a learning tool to enhance a student's 3D visualization capabilities. 

## About class `Vector`

To access the help-text of the class to know more about it, run:

```python
help(pyvectors.Vector)
```

### Class Attributes

- `Vector.i == Vector(1, 0, 0)`
- `Vector.j == Vector(0, 1, 0)`
- `Vector.k == Vector(0, 0, 1)`
- `Vector.null == Vector(0, 0, 0)`
    
These `Vectors` cannot be created through the `Vector()` constructor. Functions evaluating to these Vectors will return these predefined objects only.

### Instance Attributes

All attributes of Vector objects are read-only. Changing any of them results in unexpected behavior, which renders the object useless and prone to errors. To know the available instance attributes, run:

```python
dir(pyvectors.Vector.null)
```

To access the help-text of a method to know more about it, run (`f` is your desired Vector method):

```python
help(pyvectors.Vector.f)
```

## Update History

### Updates (0.2.2)

Minor bug fixes:
- Fixed: all math domain errors
- Fixed: issues with plotting multiple Vectors

### Updates (0.2.3)

Added methods which allow you to rotate Vectors using the Right-Hand Rule and Matrix Rotation:
- `Vector.rotate(v, yaw=x, pitch=y, roll=z)`
    > rotates the Vector by the given yaw, pitch and roll angles
- `Vector.rotate_vector(v1, v2, phi=value)`
    > rotates Vector v1 phi degrees about Vector v2
- `Vector.rotate_axis(v, axis="A", phi=value)`
    > rotates the Vector phi degrees about the positive direction of the given axis

### Updates (0.2.4)

You can now compare the magnitude of Vector objects using built-in operators < and >. Note that relational operators <= and >= will still not be defined for Vector Objects as they lead to ambiguity. (Whether to compare magnitudes or compare for actual equality)

### Updates (0.2.5)

Added new methods:
- `Vector.exp(v)`
    > returns a Vector in the direction of v whose magnitude is equal to the exponential of the magnitude of v
- `Vector.polar_repr(v)`
    > returns a string representation of v in its polar form

### Updates (0.2.6)

- Minor bug fixes:
    - Fixed: floating-point precision errors:
- Alternate constructor `Vector.FromSequence(seq)` can now accept generators and generator expressions as argument

### Updates (0.2.7)

More colors available for plotting Vectors

### Updates (0.2.8)

`Vector.random(mod)` is now a staticmethod

## Footnotes

The project does not (yet) use private/read-only class attributes available in Python where they ideally should be.

## Run

To use, execute:

```
pip install pyvectors
```

Import the class `Vector` in your project, wherever needed, using:

```python
from pyvectors import Vector
```
