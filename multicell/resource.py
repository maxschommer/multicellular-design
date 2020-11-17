from multicell.config import MICROMETER


class Resource():
    def __init__(self, value: float, units: str):
        self.value = value
        self.units = units


class Metal(Resource):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class Energy(Resource):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class Silicon(Resource):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class Oxygen(Resource):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class Hydrogen(Resource):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
